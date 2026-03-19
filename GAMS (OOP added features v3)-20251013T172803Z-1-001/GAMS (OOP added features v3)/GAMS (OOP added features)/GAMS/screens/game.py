# screens/game.py

import pygame
import time
import sys
import os
import cv2  # Re-add cv2 for video playback

from utils import resources_path, pretty_label, command_converter, get_pygame_window_pos
from config import PENALTY_RATE, TMAX, ACCEPT_N, TRAINING_ACTIONS, THRESHOLD, ACTIONS, SEQUENCE_LENGTH
from pose_detector import PoseDetector  # Import the new PoseDetector class
from ui_manager import UIManager  # Import the new UIManager class
from audio_manager import AudioManager  # Import the new AudioManager class
from video_manager import VideoManager  # Import the new VideoManager class


class Game:
    """
    Main game screen class for the training and assessment mode in GAMS.
    Manages the training sequence, pose detection, scoring, UI updates, audio/video playback,
    and user interactions. Integrates PoseDetector for ML inference, UIManager for UI,
    AudioManager for sounds, and VideoManager for videos.
    """
    # Reference config values as class attributes for easy access
    PENALTY_RATE = PENALTY_RATE  # Penalty per second for scoring
    TMAX = TMAX  # Max time allowed for penalty calculation
    ACCEPT_N = ACCEPT_N  # Consecutive frames needed to accept a detection
    ACTIONS = ACTIONS  # Core actions for model prediction
    TRAINING_ACTIONS = TRAINING_ACTIONS  # Extended actions for training sequence

    def __init__(self, win_size, win, audio=None, state=None):
        """
        Initializes the Game screen with window size, Pygame window, optional shared audio manager,
        and game state. Sets up all managers and initializes game state variables.

        Args:
            win_size (tuple): Window dimensions (width, height).
            win (pygame.Surface): Pygame window surface.
            audio (AudioManager, optional): Shared audio manager instance.
            state (GameState, optional): Shared game state object.
        """
        self.win = win  # Pygame window surface
        self.state = state  # Reference to shared game state

        self.init_scale(win_size)  # Calculate scaling factor

        # Initialize core managers
        self.pose_detector = PoseDetector(win_size, win)  # Handles pose detection and camera
        self.ui_manager = UIManager(win_size, win, self.scale, self.pose_detector)  # Manages UI elements
        self.audio_manager = audio if audio else AudioManager()  # Audio playback manager
        self.video_manager = VideoManager(win_size, win, self.pose_detector, self.scale, audio)  # Video playback manager

        # Expose buttons from ui_manager for backward compatibility with app_controller
        self.buttons = self.ui_manager.buttons

        self.actions = TRAINING_ACTIONS  # List of actions in training sequence

        # Training state flags
        self.training_started = False  # Whether training has begun
        self.instruction = "None"  # Current instruction text
        self.current_action = 0  # Index of current action in sequence
        self.button_states = {  # States for UI buttons
            "END TRAINING": False,
            "START": False
        }
        self.signal_detected = False  # Whether current action signal was detected
        self.assessment_stage = False  # Flag for assessment mode (not used in this code)

        # Scoring state variables
        self.scores = {}  # Dictionary to store scores for each action
        self.t_prompt = None  # Timestamp when instruction was prompted
        self.t_prompt_end = None  # Timestamp when instruction audio ends
        self.accept_counter = 0  # Counter for consecutive correct detections
        self.accepted_for_action = False  # Whether current action has been accepted

        # User input modal state (for name/section entry)
        self.user_input_active = False  # Whether modal is active
        self.user_input_pending_start = False  # Whether start is pending after input
        self._name_text = ""  # Current name input
        self._section_text = ""  # Current section input
        self._active_field = "name"  # Which field is active for input

        # Warning system state for audio feedback on incorrect actions
        self.last_warning_time = None  # Last time warning was played
        self.last_wrong_time = None  # Start time of current wrong streak
        self.warning_played = False  # Whether warning has been played
        self.first_warning_played = False  # Whether first warning in streak was played
        self.waiting_for_interval = False  # Whether waiting for interval between warnings
        self.audio_end_time = None  # Time when current audio ends

    def init_scale(self, win_size):
        """
        Calculates the scaling factor based on base resolution (640x360) to ensure
        UI elements scale proportionally across different window sizes.

        Args:
            win_size (tuple): Current window dimensions.
        """
        base_width, base_height = 640, 360  # Base resolution for scaling
        scale_w = win_size[0] / base_width  # Width scaling ratio
        scale_h = win_size[1] / base_height  # Height scaling ratio
        self.scale = min(scale_w, scale_h)  # Use minimum to maintain aspect ratio

    # Detection & Scoring Utilities
    @staticmethod
    def _clamp(x, lo=0.0, hi=100.0):
        """
        Clamps a value between low and high bounds.

        Args:
            x (float): Value to clamp.
            lo (float): Lower bound.
            hi (float): Upper bound.

        Returns:
            float: Clamped value.
        """
        return max(lo, min(hi, x))

    def _mark_prompt(self):
        """
        Marks the start of a new action prompt, resetting detection counters.
        Called when moving to a new action in the training sequence.
        """
        self.t_prompt = time.perf_counter()  # Record current time as prompt start
        self.accept_counter = 0  # Reset consecutive detection counter
        self.accepted_for_action = False  # Reset acceptance flag
        self.signal_detected = False  # Reset detection flag

    def _maybe_accept_current(self):
        """
        Checks if the current action should be accepted based on pose detection.
        Handles scoring, warnings for incorrect actions, and triggers audio/video on success.
        Only runs if training is started and not all actions are complete.
        """
        if not self.training_started:  # Skip if training not started
            return
        if self.current_action >= len(self.actions):  # Skip if sequence complete
            return

        required = self.actions[self.current_action]  # Required action for current step
        current_time = time.perf_counter()  # Current timestamp

        # Correct signal resets everything and increments counter
        if self.pose_detector.signal == required and self.pose_detector.confidence >= THRESHOLD:
            self.accept_counter += 1  # Increment consecutive correct detections
            self.last_wrong_time = None  # Reset wrong streak
            self.first_warning_played = False  # Reset warning flags
            self.waiting_for_interval = False
            self.audio_end_time = None

        else:
            self.accept_counter = 0  # Reset counter on incorrect signal

            # Start wrong streak tracking
            if self.last_wrong_time is None:
                self.last_wrong_time = current_time  # Mark start of wrong streak
                self.first_warning_played = False
                self.waiting_for_interval = False
                self.audio_end_time = None

            wrong_duration = current_time - self.last_wrong_time  # Duration of wrong streak

            # First warning after 5 seconds of wrong signals
            if not self.first_warning_played and wrong_duration >= 5.0:
                self.play_warning_audio()  # Play warning sound
                self.first_warning_played = True
                self.waiting_for_interval = True  # Start waiting for audio to finish
                self.audio_end_time = None  # Will be set when audio stops

            # Subsequent warnings every 5 seconds after first
            elif self.first_warning_played:
                if self.waiting_for_interval:
                    # Wait until current audio finishes
                    if not pygame.mixer.get_busy() and self.audio_end_time is None:
                        self.audio_end_time = current_time  # Mark audio end time
                else:
                    # Check if 5-second interval has passed since audio ended
                    interval = 5.0
                    if current_time - self.audio_end_time >= interval:
                        self.play_warning_audio()  # Play next warning
                        self.waiting_for_interval = True
                        self.audio_end_time = None

                # Reset waiting flag once audio end time is set
                if self.audio_end_time is not None:
                    self.waiting_for_interval = False

        # Normal acceptance flow: if enough consecutive correct detections
        if (not self.accepted_for_action) and (self.accept_counter >= self.ACCEPT_N):
            t_correct = current_time  # Time of correct detection
            if self.t_prompt is None:
                self.t_prompt = t_correct  # Fallback if prompt time not set
            time_to_correct = max(0.0, t_correct - self.t_prompt)  # Time taken to respond
            time_to_correct = min(time_to_correct, self.TMAX)  # Cap at max time
            score = 100.0 - (self.PENALTY_RATE * time_to_correct)  # Calculate score with penalty
            score = self._clamp(score)  # Clamp to 0-100
            label = pretty_label(required)  # Get pretty label for action
            self.scores[label] = score  # Store score

            self.accepted_for_action = True  # Mark action as accepted
            self.signal_detected = True  # Set detection flag

            # Special handling for chocks_inserted: play video
            if required == "chocks_inserted":
                self.play_chocksinserted_video(get_pygame_window_pos())

            self.play_detection_audio()  # Play success audio

    # Update / Inference
    def update_frame(self):
        """
        Main update loop: processes pose detection, updates UI, and checks for action acceptance.
        Called every frame to keep the game state current.
        """
        # self.pose_detector.visibility_toggle = self.visibility_toggle # Ensure pose detector knows visibility state (commented out)
        self.pose_detector.update_frame()  # Update camera feed and pose detection
        self.ui_manager.update_prediction_text()  # Refresh UI with latest signal/confidence

        # Open the detection window only after instruction audio ends and no signal detected yet
        if self.training_started and (not pygame.mixer.get_busy()) and (not self.signal_detected):
            if self.t_prompt is None:  # Set prompt time if not set
                self.t_prompt = time.perf_counter()
                self.accept_counter = 0
                self.accepted_for_action = False

            self._maybe_accept_current()  # Check for action acceptance

    # Draw
    def draw(self):
        """
        Draws the current game screen using the UI manager.
        Includes camera feed, UI panels, predictions, and guide animations.
        """
        self.ui_manager.draw(self.training_started, self.current_action)

    # --- User Input Modal (scaffold) ---
    def open_user_input(self, prefill_name=None, prefill_section=None, pending_start=False):
        """
        Opens the user input modal for entering name and section before starting training.

        Args:
            prefill_name (str, optional): Prefill name field.
            prefill_section (str, optional): Prefill section field.
            pending_start (bool): Whether to start training after input.
        """
        self.ui_manager.open_user_input(prefill_name, prefill_section, pending_start)
        # Sync local state with UI manager
        self.user_input_active = self.ui_manager.user_input_active
        self.user_input_pending_start = self.ui_manager.user_input_pending_start
        self._name_text = self.ui_manager._name_text
        self._section_text = self.ui_manager._section_text
        self._active_field = self.ui_manager._active_field

    def handle_user_input_event(self, event):
        """
        Handles keyboard/mouse events for the user input modal.

        Args:
            event (pygame.Event): The event to process.

        Returns:
            bool or None: Result of input handling (e.g., submission success).
        """
        result = self.ui_manager.handle_user_input_event(event, self.state)
        # Sync local state after handling
        self.user_input_active = self.ui_manager.user_input_active
        self._name_text = self.ui_manager._name_text
        self._section_text = self.ui_manager._section_text
        self._active_field = self.ui_manager._active_field
        return result

    # Audio
    def play_detection_audio(self):
        """
        Plays the detection audio for the current action (success sound).
        """
        self.audio_manager.play_detection(self.actions[self.current_action])

    def play_instruction_audio(self):
        """
        Plays the instruction audio for the current action and sets timing variables.
        """
        action = self.actions[self.current_action]
        self.audio_manager.play_instruction(action)  # Play instruction sound
        self.t_prompt = None  # Reset prompt time (will be set after audio)
        try:
            # Calculate when audio will end
            self.t_prompt_end = time.perf_counter() + float(self.audio_manager.instruction_audio[action].get_length())
        except Exception:
            self.t_prompt_end = None  # Fallback if length unavailable

    def play_bookends_audio(self, name):
        """
        Plays bookend audio (e.g., introduction or ending).

        Args:
            name (str): Name of the audio file (e.g., "introduction").
        """
        self.audio_manager.play_bookend(name)

    def play_warning_audio(self):
        """
        Plays warning audio for incorrect actions.
        """
        self.audio_manager.play_warning()

    def stop_current_audio(self):
        """
        Stops any currently playing audio.
        """
        self.audio_manager.stop()

    # Videos
    def play_introduction_video(self):
        """
        Plays the introduction video with accompanying audio.
        """
        self.play_bookends_audio("introduction")  # Play intro audio
        self.video_manager.play_introduction_video()  # Play intro video

    def play_chocksinserted_video(self, pygamewin_pos):
        """
        Plays the chocks inserted video at the specified window position.

        Args:
            pygamewin_pos (tuple): Position of the Pygame window (x, y).
        """
        self.video_manager.play_chocksinserted_video(pygamewin_pos)

    # Buttons
    def button_over_detection(self, mouse_pos):
        """
        Updates button hover states based on mouse position.

        Args:
            mouse_pos (tuple): Mouse position (x, y).
        """
        self.ui_manager.button_over_detection(mouse_pos)

    def visibilitybtn_over_detection(self, mouse_pos):
        """
        Updates visibility button hover state.

        Args:
            mouse_pos (tuple): Mouse position (x, y).
        """
        self.ui_manager.visibilitybtn_over_detection(mouse_pos)

    def button_down_detection(self, mouse_pos):
        """
        Detects button clicks and returns the clicked button label.

        Args:
            mouse_pos (tuple): Mouse click position (x, y).

        Returns:
            str or None: Label of clicked button, or None.
        """
        return self.ui_manager.button_down_detection(mouse_pos)

    def visibilitybtn_down_detection(self, mouse_pos):
        """
        Detects visibility button clicks.

        Args:
            mouse_pos (tuple): Mouse click position (x, y).

        Returns:
            str or None: "VISIBILITY" if clicked, else None.
        """
        return self.ui_manager.visibilitybtn_down_detection(mouse_pos)

    # Quit
    def release(self):
        """
        Releases resources, such as camera capture, when exiting.
        """
        self.pose_detector.release()

    # Compatibility methods for app_controller
    def init_opencv(self, win_size):
        """
        Initializes OpenCV camera capture (compatibility method).

        Args:
            win_size (tuple): Window size.
        """
        self.pose_detector.init_opencv(win_size)

    def init_panels(self, win_size):
        """
        Initializes UI panels (compatibility method).

        Args:
            win_size (tuple): Window size.
        """
        self.ui_manager.init_panels(win_size)

    def setup_visual_instruction_text(self, instruction):
        """
        Sets up visual instruction text in UI (compatibility method).

        Args:
            instruction (str): Instruction text.
        """
        self.ui_manager.setup_visual_instruction_text(instruction)

    def setup_visibility_button(self):
        """
        Sets up the visibility toggle button (compatibility method).
        """
        self.ui_manager.setup_visibility_button("X")  # Default to "X"

    @property
    def visibility_toggle(self):
        """
        Property for visibility toggle state (placeholder).

        Returns:
            str: "X" (default state).
        """
        return "X"  # Placeholder, can be updated if needed

    @visibility_toggle.setter
    def visibility_toggle(self, value):
        """
        Setter for visibility toggle (placeholder).

        Args:
            value: New value (ignored).
        """
        pass  # Placeholder
