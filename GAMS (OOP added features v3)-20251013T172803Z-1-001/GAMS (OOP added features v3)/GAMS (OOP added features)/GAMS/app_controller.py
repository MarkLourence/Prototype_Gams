import os
import sys
import pygame

from config import WIN_SIZE, WIN_CAPTION
from utils import resource_path, command_converter, send_command
from screens.menu import Menu
from screens.game import Game
from screens.game_over import GameOver
from screens.realtime import RealTime
from audio_manager import AudioManager
from game_state import GameState


class AppController:
    """
    Central controller for the GAMS application, managing the main application flow
    and coordinating between different screens (Menu, Game, GameOver, RealTime).
    Handles initialization, event loops, state transitions, and resource management.
    Implements the main game loop architecture with separate loops for each screen state.
    """

    def __init__(self):
        """
        Initializes the AppController with None placeholders for screens and state.
        Actual initialization occurs in bootstrap() to allow for proper setup order.
        """
        self.win = None  # Pygame window surface
        self.menu = None  # Menu screen instance
        self.game = None  # Game screen instance
        self.gameover = None  # GameOver screen instance
        self.realtime = None  # RealTime screen instance
        self.state = None  # Shared game state instance

    def bootstrap(self):
        """
        Performs initial application setup including Pygame initialization,
        window creation, icon loading, and instantiation of core components.
        Sets up shared resources like audio manager and game state.
        """
        # Initialize Pygame modules
        pygame.init()
        pygame.mixer.init()

        # Create resizable window with configured size and caption
        self.win = pygame.display.set_mode(WIN_SIZE, pygame.RESIZABLE)
        pygame.display.set_caption(WIN_CAPTION)

        # Load and set application icon if available
        icon_path = resource_path("resources\\icon.png")
        if os.path.exists(icon_path):
            icon = pygame.image.load(icon_path)
            pygame.display.set_icon(icon)

        # Initialize menu screen
        self.menu = Menu(WIN_SIZE)

        # Initialize shared audio manager and pre-load static audio
        self.audio_manager = AudioManager()
        # Defer action-specific loads until Game/RealTime are created
        self.audio_manager.load_bookends()
        self.audio_manager.load_warning()

        # Initialize shared game state
        self.state = GameState()

    # --- Main Event Loops ---
    def menu_loop(self):
        """
        Main menu loop handling user navigation between different application modes.
        Processes events for quitting, window resizing, mouse interactions, and
        transitions to game or real-time modes. Manages popup states and lazy loading
        of heavy components like OpenCV and ML models.
        """
        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

                elif event.type == pygame.VIDEORESIZE:
                    new_width = max(640, event.w)
                    new_height = max(360, event.h)
                    new_size = (new_width, new_height)

                    pygame.display.set_mode(new_size, pygame.RESIZABLE)

                    was_popup_active = self.menu.popup_active
                    start_button_state = self.menu.buttons["TRAINING & ASSESSMENT"][1]
                    start_button_state = self.menu.buttons["REAL-TIME"][1]

                    self.menu.init_scale(new_size)
                    self.menu.init_menu(new_size)
                    self.menu.init_popup(new_size)
                    self.menu.init_loading(new_size)

                    if self.menu.game_initialized and self.game is not None:
                        self.game.init_scale(new_size)
                        self.game.init_opencv(new_size)
                        self.game.init_panels(new_size)

                    self.menu.popup_active = was_popup_active
                    self.menu.buttons["TRAINING & ASSESSMENT"][1] = start_button_state
                    self.menu.buttons["REAL-TIME"][1] = start_button_state

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if self.menu.popup_active:
                        if self.menu.popupbutton_down_detection(mouse_pos):
                            self.menu.popup_active = False

                    btn_label = self.menu.menubutton_down_detection(mouse_pos)
                    if btn_label == "CONNECT TO PROTOTYPE":
                        if not self.menu.game_initialized:
                            self.menu.game_loading = True
                        self.menu.popup_active = True
                    elif btn_label == "REAL-TIME":
                        run = False
                        self.realtime_loop()
                    elif btn_label == "TRAINING & ASSESSMENT":
                        run = False
                        self.game.play_introduction_video()
                        self.game_loop()

            mouse_pos = pygame.mouse.get_pos()
            self.menu.menubutton_over_detection(mouse_pos)
            self.menu.draw(self.win)

            if self.menu.game_loading and not self.menu.game_initialized:
                global cv2, load_model, mp
                import cv2  # noqa: F401
                from keras._tf_keras.keras.models import load_model  # noqa: F401
                import mediapipe as mp  # noqa: F401

                current_winsize = pygame.display.get_surface().get_size()
                self.game = Game(current_winsize, self.win, audio=self.audio_manager, state=self.state)
                self.realtime = RealTime(current_winsize, audio=self.audio_manager)
                # Load action-dependent audio now that actions are known
                self.audio_manager.load_detection_audio(self.game.actions)
                self.audio_manager.load_instruction_audio(self.game.actions)
                self.menu.game_initialized = True
                self.menu.game_loading = False
                self.menu.buttons["TRAINING & ASSESSMENT"][1] = True
                self.menu.buttons["REAL-TIME"][1] = True

        pygame.quit()
        sys.exit()

    def game_loop(self):
        """
        Main game loop for training and assessment mode.
        Handles training progression, user input collection, signal detection,
        command sending, and transitions to game over screen.
        Manages complex state interactions between training phases and user interactions.
        """
        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

                elif event.type == pygame.VIDEORESIZE:
                    new_width = max(640, event.w)
                    new_height = max(360, event.h)
                    new_size = (new_width, new_height)

                    pygame.display.set_mode(new_size, pygame.RESIZABLE)
                    self.game.init_scale(new_size)
                    self.game.init_opencv(new_size)
                    self.game.init_panels(new_size)

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()

                    # If user input modal is open, route to it only
                    if getattr(self.game, 'user_input_active', False):
                        self.game.handle_user_input_event(event)
                    else:
                        btn_label = self.game.button_down_detection(mouse_pos)
                        if btn_label == "START":
                            # Require identity first
                            if not (self.state.user_name and self.state.user_section):
                                self.game.open_user_input(self.state.user_name, self.state.user_section, pending_start=True)
                            else:
                                self.game.training_started = True
                                self.game.instruction = self.game.actions[self.game.current_action]
                                self.game.setup_visual_instruction_text(self.game.instruction)
                                self.game.play_instruction_audio()

                                self.game.buttons["START"][1] = False
                                self.game.buttons["END TRAINING"][1] = True
                                self.game.button_states["START"] = False
                                self.game.button_states["END TRAINING"] = True

                        elif btn_label == "END TRAINING":
                            self.game.training_started = False
                            self.game.instruction = "None"
                            self.game.current_action = 0

                            self.game.setup_visual_instruction_text(self.game.instruction)
                            pygame.mixer.stop()

                            self.game.buttons["START"][1] = True
                            self.game.buttons["END TRAINING"][1] = False
                            self.game.button_states["START"] = True
                            self.game.button_states["END TRAINING"] = False

                    btn_label = self.game.visibilitybtn_down_detection(mouse_pos)
                    if btn_label == "VISIBILITY":
                        self.game.visibility_toggle = "+" if self.game.visibility_toggle == "X" else "X"
                        self.game.setup_visibility_button()

                elif event.type == pygame.KEYDOWN:
                    if getattr(self.game, 'user_input_active', False):
                        submitted = self.game.handle_user_input_event(event)
                        if submitted and getattr(self.game, 'user_input_pending_start', False):
                            self.game.user_input_pending_start = False
                            self.game.training_started = True
                            self.game.instruction = self.game.actions[self.game.current_action]
                            self.game.setup_visual_instruction_text(self.game.instruction)
                            self.game.play_instruction_audio()

                            self.game.buttons["START"][1] = False
                            self.game.buttons["END TRAINING"][1] = True
                            self.game.button_states["START"] = False
                            self.game.button_states["END TRAINING"] = True
                    else:
                        if event.key == pygame.K_SPACE:
                            self.game.stop_current_audio()
                            if not self.game.button_states["START"] and not self.game.training_started:
                                self.game.buttons["START"][1] = True
                                self.game.buttons["END TRAINING"][1] = False
                                self.game.button_states["START"] = True
                                self.game.button_states["END TRAINING"] = False

            if not self.game.assessment_stage:
                if self.game.training_started:
                    if not pygame.mixer.get_busy():
                        if self.game.signal_detected:
                            command = command_converter(self.game.actions[self.game.current_action])
                            if command:
                                send_command(command)

                            self.game.current_action += 1
                            self.game.accepted_for_action = False

                            if self.game.current_action >= len(self.game.actions):
                                self.game.assessment_stage = True
                                self.game.training_started = False

                                current_winsize = pygame.display.get_surface().get_size()
                                self.gameover = GameOver(current_winsize, self.game.scores, self.state)

                                run = False
                                self.gameover_loop()
                            else:
                                self.game.instruction = self.game.actions[self.game.current_action]
                                self.game.setup_visual_instruction_text(self.game.instruction)
                                self.game.play_instruction_audio()
                                self.game.signal_detected = False
                else:
                    if not pygame.mixer.get_busy():
                        self.game.buttons["START"][1] = True
                        self.game.button_states["START"] = True

            self.game.update_frame()
            self.game.draw()

            mouse_pos = pygame.mouse.get_pos()
            self.game.button_over_detection(mouse_pos)

            pygame.display.update()

        self.game.release()
        pygame.quit()
        sys.exit()

    def gameover_loop(self):
        """
        Game over screen loop displaying results and offering options to exit or retake.
        Handles button interactions for navigation back to menu or restarting the mission.
        Manages cleanup and reinitialization of game state for retries.
        """
        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    btn_label = self.gameover.button_down_detection(mouse_pos)
                    if btn_label == "Exit":
                        run = False

                        current_winsize = pygame.display.get_surface().get_size()
                        self.game.release()

                        self.game.__init__(current_winsize, self.win, audio=self.audio_manager, state=self.state)
                        # Centralized reset
                        self.state.reset_for_menu()
                        self.game.buttons["START"][1] = False
                        self.game.buttons["END TRAINING"][1] = False
                        self.game.button_states["START"] = False
                        self.game.button_states["END TRAINING"] = False
                        self.game.setup_visual_instruction_text(self.state.instruction)

                        self.menu.popup_active = False

                        self.menu_loop()

                    elif btn_label == "Retake Mission":
                        current_winsize = pygame.display.get_surface().get_size()
                        self.game.release()

                        self.game.__init__(current_winsize, self.win, audio=self.audio_manager, state=self.state)
                        # Centralized reset
                        self.state.reset_for_menu()
                        self.game.buttons["START"][1] = False
                        self.game.buttons["END TRAINING"][1] = False
                        self.game.button_states["START"] = False
                        self.game.button_states["END TRAINING"] = False
                        self.game.setup_visual_instruction_text(self.state.instruction)

                        # Prompt user to confirm or change identity
                        self.game.open_user_input(self.state.user_name, self.state.user_section, pending_start=False)

                        run = False
                        self.game_loop()

                    btn_label = self.game.visibilitybtn_down_detection(mouse_pos)
                    if btn_label == "VISIBILITY":
                        self.game.visibility_toggle = "+" if self.game.visibility_toggle == "X" else "X"
                        self.game.setup_visibility_button()

            if not pygame.mixer.get_busy():
                self.game.buttons["START"][1] = True
                self.game.button_states["START"] = True

            mouse_pos = pygame.mouse.get_pos()
            self.game.buttons["START"][1] = False
            self.game.buttons["END TRAINING"][1] = False
            self.game.button_states["START"] = False
            self.game.button_states["END TRAINING"] = False

            self.game.visibilitybtn_over_detection(mouse_pos)
            self.gameover.button_over_detection(mouse_pos)

            self.game.update_frame()
            self.game.draw()
            self.gameover.draw(self.win)
            pygame.display.update()

        pygame.quit()
        sys.exit()

    def realtime_loop(self):
        """
        Real-time mode loop for continuous gesture recognition and command sending.
        Handles navigation back to menu, real-time prediction display, and visibility toggles.
        Manages continuous video processing and command transmission with timeout handling.
        """
        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

                elif event.type == pygame.VIDEORESIZE:
                    new_width = max(640, event.w)
                    new_height = max(360, event.h)
                    new_size = (new_width, new_height)

                    pygame.display.set_mode(new_size, pygame.RESIZABLE)

                    self.realtime.init_scale(new_size)
                    self.realtime.init_opencv(new_size)
                    self.realtime.init_panel(new_size)
                    self.realtime.init_prediction_text()
                    self.realtime.init_buttons()
                    self.realtime.init_visibility_button()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()

                    btn_label = self.realtime.button_down_detection(mouse_pos)
                    if btn_label == "BACK TO MENU":
                        run = False
                        self.menu_loop()
                    elif btn_label == "START REAL-TIME":
                        self.realtime.realtime_started = True

                    btn_label = self.realtime.visibilitybtn_down_detection(mouse_pos)
                    if btn_label == "VISIBILITY":
                        self.realtime.visibility_toggle = "+" if self.realtime.visibility_toggle == "X" else "X"
                        self.realtime.init_visibility_button()

            if self.realtime.realtime_started:
                command = command_converter(self.realtime.signal)
                if command:
                    send_command(command, timeout=0.1)

            self.realtime.update_frame()
            self.realtime.draw(self.win)

            mouse_pos = pygame.mouse.get_pos()
            self.realtime.button_over_detection(mouse_pos)

            pygame.display.update()

        # Clean up realtime resources
        if hasattr(self, 'realtime') and self.realtime:
            self.realtime.cleanup()
        
        pygame.quit()
        sys.exit()

    def run(self):
        """
        Entry point for the application. Initializes the app and starts the main menu loop.
        This method orchestrates the entire application lifecycle from bootstrap to execution.
        """
        self.bootstrap()
        self.menu_loop()
