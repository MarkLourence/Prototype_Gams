# Import pygame for UI rendering and event handling
import pygame
# Import os for file system operations (checking directories, file paths)
import os
# Import cv2 for potential video playback in UI (though not heavily used in this file)
import cv2 # For video playback in UI, if any

# Import utility functions: resources_path for asset locations, pretty_label for text formatting, get_pygame_window_pos for window positioning
from utils import resources_path, pretty_label, get_pygame_window_pos
# Import TRAINING_ACTIONS from config to define available actions for guide videos and progress tracking
from config import TRAINING_ACTIONS # For guide videos and progress bar

# Define the UIManager class responsible for managing all user interface elements and rendering
class UIManager:
    # Initialize the UIManager with window dimensions, pygame window surface, scaling factor, and pose detector reference
    def __init__(self, win_size, win, scale, pose_detector):
        self.win_size = win_size  # Store the window size tuple (width, height)
        self.win = win  # Reference to the main pygame window surface for drawing
        self.scale = scale  # Scaling factor for UI elements to adapt to different window sizes
        self.pose_detector = pose_detector  # Reference to pose detector for accessing camera frame and position data

        # Initialize UI Rects and Surfaces - these define the layout areas and rendered text/graphics
        self.rp_top_rect = None  # Rectangle for right panel top section (prediction display)
        self.rp_bottom_rect = None  # Rectangle for right panel bottom section (buttons)
        self.lp_top_rect = None  # Rectangle for left panel top section (instructions)
        self.lp_bottom_rect = None  # Rectangle for left panel bottom section (progress bar)
        self.prediction_text_surfaces = []  # List of tuples containing prediction text surfaces and their positions
        self.predicted_label_surface = None  # Surface for the current predicted gesture label
        self.predicted_probability_surface = None  # Surface for the prediction confidence percentage
        self.buttons = {}  # Dictionary mapping button labels to their properties (hover state, enabled state, surface, position, rect)
        self.visibility_button = None  # Special button for toggling pose landmark visibility
        self.visinstr_title_text = None  # Surface for the "INSTRUCTIONS:" title text
        self.visinstr_title_pos = None  # Position tuple for the instructions title
        self.visinstr_instr_text = None  # Surface for the current instruction text
        self.visinstr_instr_pos = None  # Position tuple for the instruction text
        self.progress_rects = []  # List of rectangles representing progress bar segments
        self.progress_font = None  # Font object for progress percentage text
        self.progress_text_pos = None  # Position for the progress percentage text
        self.pilots_pov_text = None  # Surface for "From Pilot's Point of View" text
        self.pilots_pov_rect = None  # Rectangle for positioning the pilot's POV text

        # Guide Videos - configuration for animated gesture demonstration videos
        self.guide_position = [0, 0]  # Base position offset for guide video placement
        self.guide_videos = {}  # Dictionary mapping action names to (frames_list, position_tuple)
        # Configuration dictionary defining offset, size scaling, and frame delay for each action's guide video
        self.action_configs = {
            "straight_ahead": {"offset": (64, 10), "size": 450 * 0.95, "frame_delay": 5},  # Straight ahead gesture video settings
            "turn_left": {"offset": (43, 10), "size": 450 * 0.95, "frame_delay": 5},  # Left turn gesture video settings
            "turn_right": {"offset": (64, 10), "size": 450 * 0.95, "frame_delay": 5},  # Right turn gesture video settings
            "stop": {"offset": (52, 12), "size": 450 * 0.95, "frame_delay": 10},  # Stop gesture video settings (slower animation)
            "cut_engine": {"offset": (113, 55), "size": 570 * 0.95, "frame_delay": 7},  # Engine cut gesture video settings
            "start_engine": {"offset": (43, 3), "size": 430 * 0.95, "frame_delay": 5},  # Engine start gesture video settings
            "set_brakes": {"offset": (27, 2), "size": 450 * 0.95, "frame_delay": 10},  # Brake setting gesture video settings
            "chocks_inserted": {"offset": (38, -7), "size": 430 * 0.92, "frame_delay": 7},  # Wheel chocks gesture video settings
            "all_clear": {"offset": (44, 0), "size": 430 * 0.95, "frame_delay": 10},  # All clear gesture video settings
        }
        self.actions = TRAINING_ACTIONS  # List of all training actions from config
        # Extract frame delay values for each action in the same order as TRAINING_ACTIONS
        self.frame_delays = [self.action_configs[action]["frame_delay"] for action in self.actions]

        # User input modal - variables for the name/section input dialog
        self.user_input_active = False  # Flag indicating if the user input modal is currently displayed
        self.user_input_pending_start = False  # Flag indicating if training start is pending user input completion
        self._name_text = ""  # Current text entered in the name field
        self._section_text = ""  # Current text entered in the section field
        self._active_field = "name"  # Which input field currently has focus ("name" or "section")
        self._name_rect = None  # Rectangle defining the name input field area
        self._section_rect = None  # Rectangle defining the section input field area
        self._ok_rect = None  # Rectangle for the OK button in the modal
        self._cancel_rect = None  # Rectangle for the Cancel button in the modal

        self.init_panels(win_size)  # Initialize all UI panels and elements with the given window size

    # Initialize all UI panels and elements based on the current window size
    def init_panels(self, win_size):
        # Nested function to set up the basic layout rectangles for UI panels
        def setup_layout():
            cam_top_y = self.pose_detector.frame_draw_pos[1]  # Get the top Y coordinate of camera frame
            cam_bottom_y = cam_top_y + self.pose_detector.frame_draw_size[1]  # Calculate bottom Y coordinate of camera frame
            center_x = win_size[0] // 2  # Calculate center X coordinate of window
            panel_width = win_size[0] // 2  # Each panel takes half the window width
            full_height = win_size[1]  # Full window height

            # Create rectangles for the four main UI panels around the camera frame
            self.rp_top_rect = pygame.Rect(center_x, 0, panel_width, cam_top_y)  # Right panel top (prediction display)
            self.rp_bottom_rect = pygame.Rect(center_x, cam_bottom_y, panel_width, full_height - cam_bottom_y)  # Right panel bottom (buttons)
            self.lp_top_rect = pygame.Rect(0, 0, center_x, cam_top_y)  # Left panel top (instructions)
            self.lp_bottom_rect = pygame.Rect(0, cam_bottom_y, panel_width, full_height - cam_bottom_y)  # Left panel bottom (progress bar)

        # Nested function to set up the prediction text surfaces and positioning
        def setup_prediction_text():
            small_size = int(self.rp_top_rect.height * 0.3)  # Calculate small font size based on panel height
            big_size = int(self.rp_top_rect.height * 0.525)  # Calculate big font size based on panel height
            font_small = pygame.font.SysFont("Franklin Gothic Medium Condensed", small_size)  # Create small font
            font_big = pygame.font.SysFont("Franklin Gothic Medium Condensed", big_size)  # Create big font

            # Render text labels and values for signal prediction display
            label_left = font_small.render("SIGNAL PREDICTION:", True, (0, 0, 0))  # Left label text
            value_left = font_big.render(self.pose_detector.signal, True, (0, 0, 0))  # Current signal value
            label_right = font_small.render("PROBABILITY:", True, (0, 0, 0))  # Right label text
            value_right = font_big.render(f"{self.pose_detector.confidence * 100:.0f}%", True, (0, 0, 0))  # Confidence percentage

            # Get rectangles for positioning calculations
            label_left_rect = label_left.get_rect()
            value_left_rect = value_left.get_rect()
            label_right_rect = label_right.get_rect()
            value_right_rect = value_right.get_rect()

            spacing = int(2 * self.scale)  # Calculate spacing between text elements
            row_height = label_left_rect.height + value_left_rect.height + spacing  # Total height of one row
            y_start = self.rp_top_rect.centery - row_height // 2  # Center the text vertically in the panel

            margin = int(15 * self.scale)  # Calculate margin from panel edges
            left_x = self.rp_top_rect.left + margin  # Left edge position with margin
            right_x = self.rp_top_rect.right - margin  # Right edge position with margin

            # Create list of text surfaces with their positions for rendering
            self.prediction_text_surfaces = [
                (label_left, (left_x, y_start)),  # Signal prediction label
                (value_left, (left_x, y_start + label_left_rect.height + spacing)),  # Signal value
                (label_right, (right_x - label_right_rect.width, y_start)),  # Probability label
                (value_right, (right_x - value_right_rect.width, y_start + label_right_rect.height + spacing)),  # Probability value
            ]

            # Store references to the dynamic text surfaces for updates
            self.predicted_label_surface = value_left
            self.predicted_probability_surface = value_right

        # Nested function to set up the main action buttons (START and END TRAINING)
        def setup_bookend_buttons():
            button_scale = self.scale  # Allow buttons to scale up with window size
            font_size = int(16 * button_scale)  # Calculate button font size
            font = pygame.font.SysFont("Franklin Gothic Medium Condensed", font_size)  # Create button font
            labels = ["END TRAINING", "START"]  # Button labels
            surfaces = [font.render(text, True, (0, 0, 0)) for text in labels]  # Render button text surfaces

            padding_w = int(25 * button_scale)  # Horizontal padding for buttons
            padding_h = int(20 * button_scale)  # Vertical padding for buttons
            rects = [s.get_rect() for s in surfaces]  # Get text rectangles for size calculations
            width = max(r.width for r in rects) + padding_w  # Calculate button width
            height = max(r.height for r in rects) + padding_h  # Calculate button height
            y = self.rp_bottom_rect.y + int(10 * self.scale)  # Y position for buttons

            self.buttons = {}  # Initialize buttons dictionary
            for i, (label, surface, rect) in enumerate(zip(labels, surfaces, rects)):
                # Position END TRAINING on left, START on right
                x = self.rp_bottom_rect.left + int(20 * self.scale) if i == 0 else self.rp_bottom_rect.right - width - int(20 * self.scale)
                btn_rect = pygame.Rect(x, y, width, height)  # Create button rectangle
                text_pos = (x + (width - rect.width) // 2, y + (height - rect.height) // 2)  # Center text in button
                is_open = True if label == "START" else False  # START button enabled initially, END TRAINING disabled
                self.buttons[label] = [False, is_open, surface, text_pos, btn_rect]  # Store button properties: [is_hovered, is_open, text_surface, text_pos, btn_rect]

        # Nested function to load and prepare guide video frames for gesture demonstrations
        def load_guide_videos():
            for action, cfg in self.action_configs.items():  # Iterate through each action's configuration
                offset = cfg["offset"]  # Get position offset for this action
                size = cfg["size"]  # Get base size for this action

                dir_path = os.path.join(resources_path, "guide_videos", action)  # Path to action's video frames
                if not os.path.isdir(dir_path):  # Check if directory exists
                    print(f"[GuideVideos] Missing directory: {dir_path}")  # Log missing directory
                    frame_paths = []  # Empty list if directory missing
                else:
                    # Get sorted list of image files for this action
                    frame_paths = sorted(
                        [os.path.join(dir_path, f) for f in os.listdir(dir_path)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))]  # Only image files
                    )
                frames = []  # List to store processed frames

                for path in frame_paths:  # Process each frame image
                    img = pygame.image.load(path).convert()  # Load and convert image
                    if action not in ["turn_left", "set_brakes"]:  # Flip most actions horizontally for correct orientation
                        img = pygame.transform.flip(img, True, False)
                    scaled = pygame.transform.smoothscale(img, (int(size * self.scale), int(size * self.scale)))  # Scale to appropriate size
                    frames.append(scaled)  # Add processed frame to list

                x, y = self.guide_position  # Get base guide position
                x_offset, y_offset = offset  # Unpack offset values
                # Store frames and calculated position for this action
                self.guide_videos[action] = (
                    frames,
                    ((x - x_offset) * self.scale, (y - y_offset) * self.scale)  # Apply scaling to position
                )

            # Create "From Pilot's Point of View" text for specific actions
            small_size = int(16 * self.scale)  # Font size for POV text
            font_small = pygame.font.SysFont("Franklin Gothic Medium Condensed", small_size)  # Create font
            self.pilots_pov_text = font_small.render("From Pilot's Point of View", True, (0, 0, 0))  # Render text
            self.pilots_pov_rect = self.pilots_pov_text.get_rect()  # Get text rectangle
            self.pilots_pov_rect.topleft = (self.lp_top_rect.left + 2, self.lp_top_rect.bottom + 2)  # Position text

        # Nested function to set up the training progress bar
        def setup_progressbar():
            num_segments = 9  # Number of segments in progress bar (matches number of actions)
            bar_width = int(300 * self.scale)  # Total width of progress bar
            bar_height = int(15 * self.scale)  # Height of progress bar
            gap = int(1 * self.scale)  # Gap between segments

            total_width = (num_segments * bar_width // num_segments) + (gap * (num_segments - 1))  # Calculate total bar width
            start_x = self.lp_bottom_rect.centerx - total_width // 2  # Center the bar horizontally
            y = self.lp_bottom_rect.y + int(30 * self.scale)  # Y position below panel

            segment_width = bar_width // num_segments  # Width of each individual segment
            self.progress_rects = []  # List to store segment rectangles
            for i in range(num_segments):  # Create rectangle for each segment
                rect = pygame.Rect(
                    start_x + i * (segment_width + gap),  # X position with gap
                    y,  # Y position
                    segment_width,  # Width of segment
                    bar_height  # Height of segment
                )
                self.progress_rects.append(rect)  # Add to list

            font_size = int(18 * self.scale)  # Font size for progress percentage
            self.progress_font = pygame.font.SysFont("Franklin Gothic Medium Condensed", font_size)  # Create font
            self.progress_text_pos = (self.progress_rects[0].x + int(42 * self.scale), y - int(8 * self.scale))  # Position for percentage text

        # Preserve button states before re-initializing to maintain UI state during window resize
        old_button_states = {label: button[1] for label, button in self.buttons.items()} if self.buttons else {}  # Save current button enabled states
        old_visibility_open = self.visibility_button[1] if self.visibility_button else True  # Save visibility button state

        # Call setup functions in order
        setup_layout()  # Set up panel layout
        setup_prediction_text()  # Set up prediction display text
        setup_bookend_buttons()  # Set up START/END buttons
        self.setup_visibility_button("X")  # Set up visibility toggle button
        load_guide_videos()  # Load guide video frames
        setup_progressbar()  # Set up progress bar
        self.setup_visual_instruction_text("None")  # Set initial instruction text

        # Restore button states after re-initialization
        for label in self.buttons:  # Restore each button's enabled state
            if label in old_button_states:
                self.buttons[label][1] = old_button_states[label]
        if self.visibility_button:  # Restore visibility button state
            self.visibility_button[1] = old_visibility_open

    # Set up the visibility toggle button for controlling pose landmark display
    def setup_visibility_button(self, visibility_toggle):
        button_scale = min(self.scale, 1.0)  # Keep buttons at default size or smaller, not larger
        font_size = int(24 * button_scale)  # Calculate font size for button text
        font = pygame.font.SysFont("Franklin Gothic Medium Condensed", font_size)  # Create button font
        surface = font.render(visibility_toggle, True, (0, 0, 0))  # Render button text surface

        padding_w = int(25 * button_scale)  # Horizontal padding around button text
        padding_h = int(14 * button_scale)  # Vertical padding around button text
        rect = surface.get_rect()  # Get text rectangle for size calculations
        width = rect.width + padding_w  # Calculate total button width
        height = rect.height + padding_h  # Calculate total button height
        y = self.rp_bottom_rect.y + int(10 * self.scale)  # Y position for button

        x = self.rp_bottom_rect.centerx - width // 2  # Center button horizontally in bottom panel
        btn_rect = pygame.Rect(x, y, width, height)  # Create button rectangle
        text_pos = (x + (width - rect.width) // 2, y + (height - rect.height) // 2)  # Center text in button
        self.visibility_button = [False, True, surface, text_pos, btn_rect]  # Store button properties: [is_hovered, is_open, text_surface, text_pos, btn_rect]

    def setup_visual_instruction_text(self, instruction):
        small_size = int(self.lp_top_rect.height * 0.35)
        big_size = int(self.lp_top_rect.height * 0.75)

        small_font = pygame.font.SysFont("Franklin Gothic Medium Condensed", small_size)
        big_font = pygame.font.SysFont("Franklin Gothic Medium Condensed", big_size)

        title_text = small_font.render("INSTRUCTIONS:", True, (0, 0, 0))
        title_rect = title_text.get_rect()

        instruction_text = big_font.render(instruction.upper().replace("_", " "), True, (0, 0, 0))
        instruction_rect = instruction_text.get_rect()

        total_height = title_rect.height + instruction_rect.height
        start_y = self.lp_top_rect.centery - total_height // 2

        self.visinstr_title_text = title_text
        self.visinstr_title_pos = (
            self.lp_top_rect.centerx - title_rect.width // 2,
            start_y
        )

        self.visinstr_instr_text = instruction_text
        self.visinstr_instr_pos = (
            self.lp_top_rect.centerx - instruction_rect.width // 2,
            start_y + title_rect.height
        )

    def update_prediction_text(self):
        big_size = int(self.rp_top_rect.height * 0.525)
        font_big = pygame.font.SysFont("Franklin Gothic Medium Condensed", big_size)

        formatted_signal = self.pose_detector.signal.replace("_", " ").upper()
        value_left = font_big.render(formatted_signal, True, (0, 0, 0))
        value_right = font_big.render(f"{self.pose_detector.confidence * 100:.0f}%", True, (0, 0, 0))

        self.predicted_label_surface = value_left
        self.predicted_probability_surface = value_right

        spacing = int(2 * self.scale)
        small_size = int(self.rp_top_rect.height * 0.3)
        font_small = pygame.font.SysFont("Franklin Gothic Medium Condensed", small_size)
        label_left_rect = font_small.render("SIGNAL PREDICTION:", True, (0, 0, 0)).get_rect()
        label_right_rect = font_small.render("PROBABILITY:", True, (0, 0, 0)).get_rect()
        row_height = label_left_rect.height + value_left.get_rect().height + spacing
        y_start = self.rp_top_rect.centery - row_height // 2
        margin = int(15 * self.scale)
        left_x = self.rp_top_rect.left + margin
        right_x = self.rp_top_rect.right - margin

        self.prediction_text_surfaces[1] = (value_left, (left_x, y_start + label_left_rect.height + spacing))
        self.prediction_text_surfaces[3] = (value_right, (right_x - value_right.get_width(), y_start + label_right_rect.height + spacing))

    def draw(self, training_started, current_action):
        self.win.fill((255, 255, 255))
        border_width = max(1, round(2 * self.scale))

        if self.pose_detector.frame_surface:
            # Draw guide
            if training_started:
                self.draw_guide_animation(self.win, self.actions[current_action], self.frame_delays[current_action])
                for i, rect in enumerate(self.progress_rects):
                    color = (0, 200, 0) if i < current_action else (180, 180, 180)  # green if lit
                    pygame.draw.rect(self.win, color, rect)

                percentage = round((current_action / len(self.progress_rects)) * 100)
                text_surface = self.progress_font.render(f"Progress: {percentage}%", True, (0, 0, 0))
                text_rect = text_surface.get_rect(center=self.progress_text_pos)
                self.win.blit(text_surface, text_rect)

                if current_action == 2 or current_action == 3:
                    self.win.blit(self.pilots_pov_text, self.pilots_pov_rect)

            scaled_frame = pygame.transform.smoothscale(self.pose_detector.frame_surface, self.pose_detector.frame_draw_size)
            self.win.blit(scaled_frame, self.pose_detector.frame_draw_pos)

            pygame.draw.rect(self.win, (192, 192, 192), self.rp_top_rect)
            pygame.draw.rect(self.win, (119, 163, 200), self.lp_top_rect)

            for surface, pos in self.prediction_text_surfaces:
                self.win.blit(surface, pos)
            self.win.blit(self.visinstr_title_text, self.visinstr_title_pos)
            self.win.blit(self.visinstr_instr_text, self.visinstr_instr_pos)

            for is_hovered, is_open, text, text_pos, btn_rect in self.buttons.values():
                fill = (192, 192, 192) if is_hovered and is_open else \
                    (240, 240, 240) if is_open else (132, 132, 132)
                pygame.draw.rect(self.win, fill, btn_rect)
                pygame.draw.rect(self.win, (0, 0, 0), btn_rect, border_width)
                self.win.blit(text, text_pos)

            # Draw user input overlay if active (scaffold)
            if self.user_input_active:
                self._draw_user_input_overlay()

            is_hovered, is_open, text, text_pos, btn_rect = self.visibility_button
            fill = (192, 192, 192) if is_hovered and is_open else \
                    (240, 240, 240) if is_open else (132, 132, 132)
            pygame.draw.rect(self.win, fill, btn_rect)
            pygame.draw.rect(self.win, (0, 0, 0), btn_rect, border_width)
            self.win.blit(text, text_pos)

        # Also support overlay on top of visibility button area
        if self.user_input_active:
            self._draw_user_input_overlay()

    def draw_guide_animation(self, screen, action, frame_delay):
        if not hasattr(self, 'guide_frame_counters'):
            self.guide_frame_counters = {}
            self.guide_frame_timers = {}

        frames, pos = self.guide_videos[action]

        if action not in self.guide_frame_counters:
            self.guide_frame_counters[action] = 0
            self.guide_frame_timers[action] = 0

        idx = self.guide_frame_counters[action]

        screen.blit(frames[idx], pos)

        self.guide_frame_timers[action] += 1
        if self.guide_frame_timers[action] >= frame_delay:
            self.guide_frame_counters[action] = (idx + 1) % len(frames)
            self.guide_frame_timers[action] = 0

    # --- User Input Modal (scaffold) ---
    def open_user_input(self, prefill_name=None, prefill_section=None, pending_start=False):
        self.user_input_active = True
        self.user_input_pending_start = pending_start
        self._name_text = prefill_name or ""
        self._section_text = prefill_section or ""
        self._active_field = "name"

    def handle_user_input_event(self, event, state):
        if not self.user_input_active:
            return None
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_TAB:
                self._active_field = "section" if self._active_field == "name" else "name"
            elif event.key == pygame.K_BACKSPACE:
                if self._active_field == "name":
                    self._name_text = self._name_text[:-1]
                else:
                    self._section_text = self._section_text[:-1]
            elif event.key == pygame.K_RETURN:
                return self._submit_user_input(state)
            elif event.key == pygame.K_ESCAPE:
                self.user_input_active = False
            else:
                ch = getattr(event, 'unicode', '')
                if ch and ch.isprintable():
                    if self._active_field == "name":
                        self._name_text += ch
                    else:
                        self._section_text += ch
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            if hasattr(self, '_name_rect') and self._name_rect.collidepoint((mx, my)):
                self._active_field = "name"
            elif hasattr(self, '_section_rect') and self._section_rect.collidepoint((mx, my)):
                self._active_field = "section"
            elif hasattr(self, '_ok_rect') and self._ok_rect.collidepoint((mx, my)):
                return self._submit_user_input(state)
            elif hasattr(self, '_cancel_rect') and self._cancel_rect.collidepoint((mx, my)):
                self.user_input_active = False

    def _submit_user_input(self, state):
        name = self._name_text.strip()
        section = self._section_text.strip()
        if name and section:
            if state is not None:
                state.user_name = name
                state.user_section = section
            self.user_input_active = False
            return True
        return None

    def _draw_user_input_overlay(self):
        overlay = pygame.Surface(self.win.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 100))
        self.win.blit(overlay, (0, 0))

        pad = int(16 * self.scale)
        title_h = int(28 * self.scale)
        label_h = int(20 * self.scale)
        input_h = int(30 * self.scale)
        btn_h = int(26 * self.scale)
        width = int(400 * self.scale)
        win_w, win_h = self.win.get_size()
        x = (win_w - width) // 2
        y = (win_h - (title_h + label_h + input_h + label_h + input_h + btn_h + 5 * pad)) // 2

        container_rect = pygame.Rect(x, y, width, title_h + label_h + input_h + label_h + input_h + btn_h + 5 * pad)
        pygame.draw.rect(self.win, (255, 255, 255), container_rect)
        pygame.draw.rect(self.win, (0, 0, 0), container_rect, 2)

        font_title = pygame.font.SysFont("Franklin Gothic Medium Condensed", int(28 * self.scale), bold=True)
        font_label = pygame.font.SysFont("Franklin Gothic Medium Condensed", int(18 * self.scale))
        font_input = pygame.font.SysFont("Franklin Gothic Medium Condensed", int(20 * self.scale))
        font_btn = pygame.font.SysFont("Franklin Gothic Medium Condensed", int(18 * self.scale))

        title_surf = font_title.render("Enter Name and Section", True, (0, 0, 0))
        title_pos = (x + (width - title_surf.get_width()) // 2, y + pad)
        self.win.blit(title_surf, title_pos)

        cursor_y = y + pad + title_h

        name_label = font_label.render("Name:", True, (0, 0, 0))
        self.win.blit(name_label, (x + pad, cursor_y))
        cursor_y += label_h // 2

        name_rect = pygame.Rect(x + pad, cursor_y, width - 2 * pad, input_h)
        pygame.draw.rect(self.win, (255, 255, 255), name_rect)
        pygame.draw.rect(self.win, (0, 0, 0), name_rect, 2)
        name_text_surf = font_input.render(self._name_text or "", True, (0, 0, 0))
        self.win.blit(name_text_surf, (name_rect.x + pad // 2, name_rect.y + (input_h - name_text_surf.get_height()) // 2))
        if self._active_field == "name":
            pygame.draw.rect(self.win, (66, 140, 226), name_rect, 2)

        cursor_y += input_h + pad

        section_label = font_label.render("Section:", True, (0, 0, 0))
        self.win.blit(section_label, (x + pad, cursor_y))
        cursor_y += label_h // 2

        section_rect = pygame.Rect(x + pad, cursor_y, width - 2 * pad, input_h)
        pygame.draw.rect(self.win, (255, 255, 255), section_rect)
        pygame.draw.rect(self.win, (0, 0, 0), section_rect, 2)
        section_text_surf = font_input.render(self._section_text or "", True, (0, 0, 0))
        self.win.blit(section_text_surf, (section_rect.x + pad // 2, section_rect.y + (input_h - section_text_surf.get_height()) // 2))
        if self._active_field == "section":
            pygame.draw.rect(self.win, (66, 140, 226), section_rect, 2)

        cursor_y += input_h + pad

        btn_w = int(120 * self.scale)
        ok_rect = pygame.Rect(x + width - pad - btn_w, cursor_y, btn_w, btn_h)
        cancel_rect = pygame.Rect(x + pad, cursor_y, btn_w, btn_h)
        for rect, label in [(cancel_rect, "Cancel"), (ok_rect, "OK")]:
            pygame.draw.rect(self.win, (240, 240, 240), rect)
            pygame.draw.rect(self.win, (0, 0, 0), rect, 2)
            txt = font_btn.render(label, True, (0, 0, 0))
            self.win.blit(txt, (rect.x + (rect.width - txt.get_width()) // 2, rect.y + (rect.height - txt.get_height()) // 2))

        # Save rects for click detection
        self._name_rect = name_rect
        self._section_rect = section_rect
        self._ok_rect = ok_rect
        self._cancel_rect = cancel_rect

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
