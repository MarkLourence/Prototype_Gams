import pygame
import cv2
import mediapipe as mp
import numpy as np
from keras._tf_keras.keras.models import load_model
import time
import sys
import os
import threading
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import resource_path, resources_path, pretty_label, command_converter, get_pygame_window_pos
from config import SEQUENCE_LENGTH, THRESHOLD, PENALTY_RATE, TMAX, ACCEPT_N, ACTIONS, TRAINING_ACTIONS


class RealTime:
    """
    Real-time pose detection and prediction screen for GAMS.
    Provides live camera feed with MediaPipe pose estimation, ML model predictions,
    and real-time signal recognition with performance optimizations including
    background threading and frame skipping for smooth operation.
    """

    # Model path resolution for packaged executables
    MODEL_PATH = os.path.join(resources_path, "model.h5")
    if getattr(sys, 'frozen', False):
        exe_model = os.path.join(os.path.dirname(sys.executable), "model.h5")
        if os.path.exists(exe_model):
            MODEL_PATH = exe_model

    # Reference config values as class attributes for easy access
    SEQUENCE_LENGTH = SEQUENCE_LENGTH  # Number of frames in prediction sequence
    THRESHOLD = THRESHOLD  # Minimum confidence for prediction acceptance
    PENALTY_RATE = PENALTY_RATE  # Unused in real-time mode
    TMAX = TMAX  # Unused in real-time mode
    ACCEPT_N = ACCEPT_N  # Unused in real-time mode
    ACTIONS = ACTIONS  # List of possible actions for model
    TRAINING_ACTIONS = TRAINING_ACTIONS  # Extended actions (unused in real-time)
    
    def __init__(self, win_size, audio=None):
        """
        Initializes the RealTime screen with camera capture, MediaPipe pose detection,
        ML model loading, and UI setup. Starts background prediction thread for performance.

        Args:
            win_size (tuple): Window dimensions (width, height).
            audio (AudioManager, optional): Audio manager instance (unused in this class).
        """
        # Initialize camera with optimized settings for low latency
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # Performance optimization: Camera settings for reduced latency
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size for lower latency

        # Initialize MediaPipe Pose with optimized settings
        self.mp_pose = mp.solutions.pose
        # Performance optimization: Optimize MediaPipe settings for better performance
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Reduced from default 2 for better performance
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.audio = audio  # Audio manager (not used in this implementation)
        if not self.cap.isOpened():
            print("Could not open camera.")
            pygame.quit()
            sys.exit()

        # State variables
        self.realtime_started = False  # Flag for real-time mode start (unused)
        self.visibility_toggle = "X"  # Toggle for showing/hiding pose landmarks
        self.actions = TRAINING_ACTIONS  # Actions list (unused in real-time)

        # Performance optimization: Frame rate limiting
        self.clock = pygame.time.Clock()
        self.target_fps = 30  # Target FPS for smooth display

        # Performance optimization: Frame skipping for pose detection
        self.frame_skip_counter = 0
        self.frame_skip_interval = 0  # Process every frame for faster detection

        # Performance optimization: Prediction throttling
        self.last_prediction_time = 0
        self.prediction_interval = 0.03  # Predict every 30ms for faster real-time response

        # Performance optimization: Background threading for model predictions
        self.prediction_queue = deque(maxlen=10)  # Increased queue size for faster predictions
        self.prediction_lock = threading.Lock()
        self.model_thread = None
        self.stop_prediction_thread = False

        # Initialize UI and camera scaling
        self.init_scale(win_size)
        self.init_opencv(win_size)
        self.init_panel(win_size)
        self.init_prediction_text()
        self.init_buttons()
        self.init_visibility_button()

        # Start background prediction thread for non-blocking ML inference
        self.start_prediction_thread()

    def start_prediction_thread(self):
        """Start background thread for model predictions"""
        self.stop_prediction_thread = False
        self.model_thread = threading.Thread(target=self.prediction_worker, daemon=True)
        self.model_thread.start()

    def prediction_worker(self):
        """Background worker for model predictions"""
        while not self.stop_prediction_thread:
            with self.prediction_lock:
                if self.prediction_queue:
                    sequence = self.prediction_queue.popleft()
                    
                    # Model prediction
                    input_seq = np.expand_dims(np.array(sequence), axis=0)
                    probs = self.model.predict(input_seq, verbose=0)[0]
                    max_idx = np.argmax(probs)
                    confidence = float(probs[max_idx])
                    signal = self.ACTIONS[max_idx] if confidence > self.THRESHOLD else "NONE"
                    
                    # Update prediction results
                    self.signal = signal
                    self.confidence = confidence
                    self.update_prediction_text()
            time.sleep(0.001)  # Minimal delay for ultra-fast predictions

    def init_scale(self, win_size):
        base_width, base_height = 640, 360
        scale_w = win_size[0] / base_width
        scale_h = win_size[1] / base_height
        self.scale = min(scale_w, scale_h)

    def init_opencv(self, win_size):
        # Dummy frame to calculate aspect ratio
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to read dummy frame for init.")
            pygame.quit()
            sys.exit()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        dummy_surface = pygame.surfarray.make_surface(frame)

        cam_rect = dummy_surface.get_rect()
        max_width = win_size[0] // 1.75
        max_height = win_size[1]

        self.model = load_model(self.MODEL_PATH)
        self.sequence = []
        self.signal = "NONE"
        self.confidence = 0.0

        scale = min(max_width / cam_rect.width, max_height / cam_rect.height)
        new_size = (int(cam_rect.width * scale), int(cam_rect.height * scale))
        self.frame_draw_size = new_size

        pos_x = (win_size[0] - new_size[0]) // 2
        pos_y = (win_size[1] - new_size[1]) // 2
        self.frame_draw_pos = (pos_x, pos_y)

    def init_panel(self, win_size):
        background = pygame.image.load(f"{resources_path}/background.png")
        self.background = pygame.transform.scale(background, win_size)

        self.top_panel = pygame.Rect(
            self.frame_draw_pos[0], 0, 
            win_size[0] // 1.75, self.frame_draw_pos[1])
        self.bottom_panel = pygame.Rect(
            self.frame_draw_pos[0], win_size[1] - self.frame_draw_pos[1], 
            win_size[0] // 1.75, self.frame_draw_pos[1])

    def init_prediction_text(self):
        small_size = int(self.top_panel.height * 0.45)
        big_size = int(self.top_panel.height * 0.7875)
        font_small = pygame.font.SysFont("Franklin Gothic Medium Condensed", small_size)
        font_big = pygame.font.SysFont("Franklin Gothic Medium Condensed", big_size)

        label_left = font_small.render("SIGNAL PREDICTION:", True, (0, 0, 0))
        value_left = font_big.render(self.signal, True, (0, 0, 0))
        label_right = font_small.render("PROBABILITY:", True, (0, 0, 0))
        value_right = font_big.render(f"{self.confidence * 100:.0f}%", True, (0, 0, 0))

        label_left_rect = label_left.get_rect()
        value_left_rect = value_left.get_rect()
        label_right_rect = label_right.get_rect()
        value_right_rect = value_right.get_rect()

        spacing = int(2 * self.scale)
        row_height = label_left_rect.height + value_left_rect.height + spacing
        y_start = self.top_panel.centery - row_height // 2

        margin = int(15 * self.scale)
        left_x = self.top_panel.left + margin
        right_x = self.top_panel.right - margin

        self.prediction_text_surfaces = [
            (label_left, (left_x, y_start)),
            (value_left, (left_x, y_start + label_left_rect.height + spacing)),
            (label_right, (right_x - label_right_rect.width, y_start)),
            (value_right, (right_x - value_right_rect.width, y_start + label_right_rect.height + spacing)),
        ]

        self.predicted_label_surface = value_left
        self.predicted_probability_surface = value_right

    def init_buttons(self):
        font_size = int(16 * self.scale)
        font = pygame.font.SysFont("Franklin Gothic Medium Condensed", font_size)
        labels = ["BACK TO MENU", "START REAL-TIME"]
        surfaces = [font.render(text, True, (0, 0, 0)) for text in labels]

        padding_w = int(25 * self.scale)
        padding_h = int(20 * self.scale)
        rects = [s.get_rect() for s in surfaces]
        width = max(r.width for r in rects) + padding_w
        height = max(r.height for r in rects) + padding_h
        y = self.bottom_panel.y + ((self.bottom_panel.height - height) // 2)
        
        self.buttons = {}
        for i, (label, surface, rect) in enumerate(zip(labels, surfaces, rects)):
            x = self.bottom_panel.left + int(20 * self.scale) if i == 0 else self.bottom_panel.right - width - int(20 * self.scale)
            btn_rect = pygame.Rect(x, y, width, height)
            text_pos = (x + (width - rect.width) // 2, y + (height - rect.height) // 2)
            self.buttons[label] = [False, True, surface, text_pos, btn_rect]  # is_hovered, is_open, text, text_pos, btn_rect

    def init_visibility_button(self):
        font_size = int(24 * self.scale)
        font = pygame.font.SysFont("Franklin Gothic Medium Condensed", font_size)
        surface = font.render(self.visibility_toggle, True, (0, 0, 0))

        padding_w = int(25 * self.scale)
        padding_h = int(14 * self.scale)
        rect = surface.get_rect()
        width = rect.width + padding_w
        height = rect.height + padding_h
        y = self.bottom_panel.y + ((self.bottom_panel.height - height) // 2)

        x = self.bottom_panel.centerx - width // 2
        btn_rect = pygame.Rect(x, y, width, height)
        text_pos = (x + (width - rect.width) // 2, y + (height - rect.height) // 2)
        self.visibility_button = [False, True, surface, text_pos, btn_rect]

    # Draw
    def draw(self, win):
        win.blit(self.background, (0, 0))
        border_width = max(1, round(2 * self.scale))

        pygame.draw.rect(win, (192, 192, 192), self.top_panel)
        # pygame.draw.rect(win, (0, 0, 0), self.bottom_panel)

        if self.frame_surface:
            scaled_frame = pygame.transform.scale(self.frame_surface, self.frame_draw_size)
            win.blit(scaled_frame, self.frame_draw_pos)

            for surface, pos in self.prediction_text_surfaces:
                win.blit(surface, pos)

            for is_hovered, is_open, text, text_pos, btn_rect in self.buttons.values():
                fill = (192, 192, 192) if is_hovered and is_open else \
                    (240, 240, 240) if is_open else (132, 132, 132)
                pygame.draw.rect(win, fill, btn_rect)
                pygame.draw.rect(win, (0, 0, 0), btn_rect, border_width)
                win.blit(text, text_pos)

            is_hovered, is_open, text, text_pos, btn_rect = self.visibility_button
            fill = (192, 192, 192) if is_hovered and is_open else \
                    (240, 240, 240) if is_open else (132, 132, 132)
            pygame.draw.rect(win, fill, btn_rect)
            pygame.draw.rect(win, (0, 0, 0), btn_rect, border_width)
            win.blit(text, text_pos)

    # Update / Interface
    def update_frame(self):
        # Frame rate limiting for smoother performance
        self.clock.tick(self.target_fps)
        
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Performance optimization: Skip frames for pose detection
        self.frame_skip_counter += 1
        should_process_pose = (self.frame_skip_counter % (self.frame_skip_interval + 1) == 0)
        
        # MediaPipe Pose processing (only on selected frames)
        if should_process_pose:
            results = self.pose.process(frame_rgb)
        else:
            # Use previous results or create empty results
            results = type('obj', (object,), {'pose_landmarks': None})()
        # Draw landmarks and process keypoints only when pose detection is performed
        if should_process_pose:
            if results.pose_landmarks and self.visibility_toggle == "X":
                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
                )

            # Keypoint vector
            keypoints = self.extract_keypoints_full(results)
            self.sequence.append(keypoints)
            if len(self.sequence) > self.SEQUENCE_LENGTH:
                self.sequence.pop(0)

        # Heuristic overrides for All Clear and Set Brakes (sets 100% confidence)
        # Only process heuristics when pose detection was performed
        signal = ""
        if should_process_pose and results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            right_wrist_y = lm[15].y
            right_elbow_y = lm[14].y
            left_wrist_y = lm[16].y
            left_hip_y = lm[23].y
            right_shoulder_y = lm[12].y
            right_eye = lm[5].y
            
            if (right_wrist_y < right_eye and right_wrist_y < right_elbow_y and
                    left_wrist_y > left_hip_y):
                signal = "all_clear"
                self.signal = signal
                self.confidence = 1.0
                self.update_prediction_text()
            elif (right_wrist_y < right_shoulder_y and right_wrist_y < right_elbow_y and
                    left_wrist_y > left_hip_y):
                signal = "set_brakes"
                self.signal = signal
                self.confidence = 1.0
                self.update_prediction_text()

        # Model prediction if no heuristic match (only when pose detection was performed)
        if should_process_pose and signal == "" and len(self.sequence) == self.SEQUENCE_LENGTH:
            # Performance optimization: Queue prediction for background thread
            current_time = time.time()
            if current_time - self.last_prediction_time >= self.prediction_interval:
                with self.prediction_lock:
                    # Queue prediction without limit for faster processing
                    self.prediction_queue.append(list(self.sequence))
                self.last_prediction_time = current_time

        # Convert for Pygame (after drawing)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.flip(frame_rgb, 1)
        frame_rgb = np.rot90(frame_rgb)
        self.frame_surface = pygame.surfarray.make_surface(frame_rgb)

    def extract_keypoints_full(self, results):
        if not results.pose_landmarks:
            return np.zeros(33 * 3)
        return np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()

    def update_prediction_text(self):
        big_size = int(self.top_panel.height * 0.7875)
        font_big = pygame.font.SysFont("Franklin Gothic Medium Condensed", big_size)

        formatted_signal = self.signal.replace("_", " ").upper()
        value_left = font_big.render(formatted_signal, True, (0, 0, 0))
        value_right = font_big.render(f"{self.confidence * 100:.0f}%", True, (0, 0, 0))

        self.predicted_label_surface = value_left
        self.predicted_probability_surface = value_right

        spacing = int(2 * self.scale)
        small_size = int(self.top_panel.height * 0.45)
        font_small = pygame.font.SysFont("Franklin Gothic Medium Condensed", small_size)
        label_left_rect = font_small.render("SIGNAL PREDICTION:", True, (0, 0, 0)).get_rect()
        label_right_rect = font_small.render("PROBABILITY:", True, (0, 0, 0)).get_rect()
        row_height = label_left_rect.height + value_left.get_rect().height + spacing
        y_start = self.top_panel.centery - row_height // 2
        margin = int(15 * self.scale)
        left_x = self.top_panel.left + margin
        right_x = self.top_panel.right - margin

        self.prediction_text_surfaces[1] = (value_left, (left_x, y_start + label_left_rect.height + spacing))
        self.prediction_text_surfaces[3] = (value_right, (right_x - value_right.get_width(), y_start + label_right_rect.height + spacing))

    # Buttons
    def button_over_detection(self, mouse_pos):
        for button in self.buttons.values():
            button[0] = button[4].collidepoint(mouse_pos)

        self.visibilitybtn_over_detection(mouse_pos)

    def visibilitybtn_over_detection(self, mouse_pos):
        button = self.visibility_button
        button[0] = button[4].collidepoint(mouse_pos)

    def button_down_detection(self, mouse_pos):
        for label, (_, is_open, *_, btn_rect) in self.buttons.items():
            if is_open and btn_rect.collidepoint(mouse_pos):
                return label
            
        self.visibilitybtn_down_detection(mouse_pos)

    def visibilitybtn_down_detection(self, mouse_pos):
        _, is_open, *_, btn_rect = self.visibility_button
        if is_open and btn_rect.collidepoint(mouse_pos):
            return "VISIBILITY"
    
    def cleanup(self):
        """Clean up resources when exiting"""
        self.stop_prediction_thread = True
        if self.model_thread and self.model_thread.is_alive():
            self.model_thread.join(timeout=1.0)
        if hasattr(self, 'cap'):
            self.cap.release()
