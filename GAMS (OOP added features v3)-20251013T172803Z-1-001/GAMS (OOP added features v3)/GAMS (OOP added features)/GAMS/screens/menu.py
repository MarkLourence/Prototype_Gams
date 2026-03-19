# screens/menu.py

import pygame
from utils import resource_path, resources_path
from .base_screen import BaseScreen


class Menu(BaseScreen):
    """
    Main menu screen class for GAMS, inheriting from BaseScreen.
    Displays the title, background, and navigation buttons for different game modes.
    Handles popups for prototype connection and loading states.
    Manages UI scaling, button interactions, and drawing of menu elements.
    """
    def __init__(self, win_size):
        """
        Initializes the Menu screen with window size, sets up scaling,
        and initializes menu, popup, and loading UI components.

        Args:
            win_size (tuple): Current window dimensions (width, height).
        """
        super().__init__(win_size)  # Initialize BaseScreen for scaling and button helpers
        self.popup_active = False  # Flag for whether connection popup is shown
        self.game_loading = False  # Flag for whether loading screen is active
        self.game_initialized = False  # Flag for game initialization status

        self.init_menu(win_size)  # Set up main menu elements (title, buttons)
        self.init_popup(win_size)  # Set up connection success popup
        self.init_loading(win_size)  # Set up loading screen

    def init_scale(self, win_size):
        """
        Compatibility method that delegates scaling to BaseScreen.
        Ensures consistent scaling across different window sizes.

        Args:
            win_size (tuple): Window dimensions.
        """
        # keep method name for compatibility; delegate to BaseScreen
        super().init_scale(win_size)

    def init_menu(self, win_size):
        """
        Initializes the main menu elements: background image, title text, and navigation buttons.
        Calculates positions and scaling for all UI elements to center them properly.

        Args:
            win_size (tuple): Window dimensions.
        """
        # Load and scale background image to fit window
        background = pygame.image.load(f"{resources_path}/background.png")
        self.background = pygame.transform.scale(background, win_size)

        # Calculate scaled font sizes for different text elements
        garamond_size = int(54 * self.scale)
        franklinbig_size = int(112 * self.scale)
        franklinsmall_size = int(20 * self.scale)
        spacing = int(-5 * self.scale)

        # Initialize fonts for title text
        garamond = pygame.font.SysFont("Garamond", garamond_size, bold=True)
        franklingothic_big = pygame.font.SysFont("Franklin Gothic Medium Condensed", franklinbig_size, bold=False)
        franklingothic_small = pygame.font.SysFont("Franklin Gothic Medium Condensed", franklinsmall_size, bold=False)

        self.spacing = spacing  # Store spacing for layout

        # Render title text lines with different fonts and colors
        raw_texts = [
            garamond.render("GROUND", True, (22, 33, 68)),  # Dark blue
            garamond.render("AIRCRAFT", True, (22, 33, 68)),
            garamond.render("MARSHALLING", True, (22, 33, 68)),
            franklingothic_big.render("SIMULATOR", True, (66, 140, 226))  # Light blue
        ]

        self.texts = []  # List to store rendered text and positions
        # Calculate total height of all text lines for centering
        text_heights = sum(text.get_rect().height for text in raw_texts)
        total_spacing = self.spacing * (len(raw_texts) - 1) - (self.spacing * 6)
        start_y = win_size[1] // 2 - (text_heights + total_spacing) // 2  # Center vertically

        current_y = start_y
        for text in raw_texts:
            rect = text.get_rect()
            x = win_size[0] // 2 - rect.width // 2  # Center horizontally
            self.texts.append([text, (x, current_y)])  # Store surface and position
            current_y += rect.height + self.spacing
        else:
            current_y -= self.spacing * 6  # Adjust for last spacing

        # Define button labels and render their text surfaces
        button_labels = ["CONNECT TO PROTOTYPE", "REAL-TIME", "TRAINING & ASSESSMENT"]
        button_surfaces = [franklingothic_small.render(label, True, (0, 0, 0)) for label in button_labels]

        # Position buttons below the "MARSHALLING" text, centered
        marshalling_surface = self.texts[2][0]
        marshalling_rect = marshalling_surface.get_rect()
        marshalling_x = win_size[0] // 2 - marshalling_rect.width // 2
        marshalling_center_x = marshalling_x + marshalling_rect.width // 2

        # Calculate button padding and spacing
        base_padding_w, base_padding_h = 25, 20
        padding_w = int(base_padding_w * self.scale)
        padding_h = int(base_padding_h * self.scale)
        spacing = int(15 * self.scale)  # Spacing between buttons

        self.buttons = {}  # Dictionary to store button data
        # Calculate total width of all buttons for centering
        total_width = sum(surface.get_rect().width + padding_w for surface in button_surfaces) + spacing * (len(button_labels) - 1)
        start_x = marshalling_center_x - total_width // 2  # Start position for first button

        # Create button data for each button
        for idx, (label, surface) in enumerate(zip(button_labels, button_surfaces)):
            text_rect = surface.get_rect()
            x = start_x
            y = current_y

            # Create button rectangle with padding
            button_rect = pygame.Rect(
                x - (padding_w // 2),
                y - (padding_h // 2),
                text_rect.width + padding_w,
                text_rect.height + padding_h
            )

            # Center text within button
            text_x = button_rect.x + (button_rect.width - text_rect.width) // 2
            text_y = button_rect.y + (button_rect.height - text_rect.height) // 2

            # Only the first button ("CONNECT TO PROTOTYPE") starts enabled/open
            self.buttons[label] = [False, (idx == 0), surface, (text_x, text_y), button_rect]

            # Move start_x for next button
            start_x += text_rect.width + padding_w + spacing

    def init_popup(self, win_size):
        """
        Initializes the connection success popup UI elements.
        Creates text surfaces, calculates popup dimensions, and positions elements.

        Args:
            win_size (tuple): Window dimensions.
        """
        # Define popup text lines
        popup_lines = [
            "CONNECTED SUCCESSFULLY TO",
            "PROTOTYPE AIRCRAFT!",
            "",
            "PLEASE PRESS",
            "THE START BUTTON TO CONTINUE",
            "ASSESSMENT AND TRAINING."
        ]

        # Render popup text
        font = pygame.font.SysFont("Franklin Gothic Medium Condensed", int(32 * self.scale))
        self.popup_surfaces = [font.render(line, True, (0, 0, 0)) for line in popup_lines]

        # Calculate popup dimensions based on text size
        self.popup_width = max(s.get_width() for s in self.popup_surfaces) + int(40 * self.scale)
        self.popup_height = sum(s.get_height() for s in self.popup_surfaces) + int(10 * self.scale) * (len(self.popup_surfaces) - 2) + int(30 * self.scale)

        win_w, win_ht = win_size
        titlebar_height = int(22 * self.scale)
        popup_total_height = self.popup_height + titlebar_height
        vertical_center = (win_ht - popup_total_height) // 2 - (18 * self.scale)  # Center vertically with offset

        # Create titlebar rectangle
        popup_titlebar_rect = pygame.Rect(
            (win_w - self.popup_width) // 2,  # Center horizontally
            vertical_center,
            self.popup_width,
            titlebar_height
        )

        # Create main popup rectangle below titlebar
        popup_rect = pygame.Rect(
            (win_w - self.popup_width) // 2,
            vertical_center + titlebar_height,
            self.popup_width,
            self.popup_height
        )

        # Create close button (X) in titlebar
        bahnschrift = pygame.font.SysFont("Bahnschrift", int(16 * self.scale), bold=False)
        popup_closebutton_text = bahnschrift.render("X", True, (0, 0, 0))
        close_text_rect = popup_closebutton_text.get_rect()

        pad_x = int(8 * self.scale)
        pad_y = int(2 * self.scale)
        button_w = close_text_rect.width + 2 * pad_x
        button_h = close_text_rect.height + 2 * pad_y

        # Position close button in top-right of titlebar
        popup_closebutton_rect = pygame.Rect(
            popup_titlebar_rect.right - button_w - int(5 * self.scale),
            popup_titlebar_rect.centery - button_h // 2,
            button_w,
            button_h
        )

        # Center text in close button
        popup_closebutton_text_pos = (
            popup_closebutton_rect.x + (button_w - close_text_rect.width) // 2,
            popup_closebutton_rect.y + (button_h - close_text_rect.height) // 2
        )

        # Store popup elements as list for easy unpacking
        self.popup = [popup_titlebar_rect, popup_rect, popup_closebutton_text, popup_closebutton_rect, popup_closebutton_text_pos]

    def init_loading(self, win_size):
        """
        Initializes the loading screen UI elements, reusing popup dimensions for consistency.

        Args:
            win_size (tuple): Window dimensions.
        """
        titlebar_height = int(22 * self.scale)
        font_size = int(64 * self.scale)

        win_w, win_ht = win_size

        # Use same dimensions and positioning as popup for consistency
        popup_total_height = self.popup_height + titlebar_height
        vertical_center = (win_ht - popup_total_height) // 2 - (18 * self.scale)

        # Create loading titlebar
        loading_titlebar_rect = pygame.Rect(
            (win_w - self.popup_width) // 2,
            vertical_center,
            self.popup_width,
            titlebar_height
        )

        # Create loading main rectangle
        loading_rect = pygame.Rect(
            (win_w - self.popup_width) // 2,
            vertical_center + titlebar_height,
            self.popup_width,
            self.popup_height
        )

        # Render "LOADING..." text
        font = pygame.font.SysFont("Franklin Gothic Medium Condensed", font_size, bold=False)
        loading_surface = font.render("LOADING...", True, (0, 0, 0))

        # Center text in loading rectangle
        loading_x = loading_rect.x + (loading_rect.width - loading_surface.get_width()) // 2
        loading_y = loading_rect.y + (loading_rect.height - loading_surface.get_height()) // 2

        # Store loading elements
        self.loading = [loading_titlebar_rect, loading_rect, loading_surface, (loading_x, loading_y)]

    def draw(self, win):
        """
        Draws the menu screen, including background, buttons, title text, and conditional popups/loading.

        Args:
            win (pygame.Surface): Pygame window surface to draw on.
        """
        win.blit(self.background, (0, 0))  # Draw background image

        border_width = max(1, round(2 * self.scale))  # Scaled border thickness

        # Draw all menu buttons
        for button_data in self.buttons.values():
            self.draw_button(win, button_data, border_width)

        # Draw title text if no popup/loading is active
        if not self.popup_active and not self.game_loading:
            for text_surface, pos in self.texts:
                win.blit(text_surface, pos)
        elif self.game_loading:  # Draw loading screen
            loading_titlebar_rect, loading_rect, loading_surface, loading_pos = self.loading
            pygame.draw.rect(win, (255, 255, 255), loading_rect)  # White background
            pygame.draw.rect(win, (192, 192, 192), loading_titlebar_rect)  # Gray titlebar
            win.blit(loading_surface, loading_pos)  # "LOADING..." text
        elif self.popup_active:  # Draw connection popup
            popup_titlebar_rect, popup_rect, popup_closebutton_text, popup_closebutton_rect, popup_closebutton_text_pos = self.popup
            pygame.draw.rect(win, (255, 255, 255), popup_rect)  # White background
            pygame.draw.rect(win, (192, 192, 192), popup_titlebar_rect)  # Gray titlebar
            pygame.draw.rect(win, (162, 162, 162), popup_closebutton_rect)  # Light gray close button
            win.blit(popup_closebutton_text, popup_closebutton_text_pos)  # "X" text

            # Draw popup text lines, centered and spaced
            cursor_y = popup_rect.y + int(10 * self.scale)
            for idx, surf in enumerate(self.popup_surfaces):
                x = popup_rect.x + (self.popup_width - surf.get_width()) // 2  # Center horizontally
                win.blit(surf, (x, cursor_y))
                if idx < len(self.popup_surfaces) - 1:
                    cursor_y += surf.get_height() + int(10 * self.scale)  # Move down for next line

        pygame.display.update()  # Update the display

    def menubutton_down_detection(self, mouse_pos):
        """
        Detects which menu button was clicked based on mouse position.

        Args:
            mouse_pos (tuple): Mouse click position (x, y).

        Returns:
            str or None: Label of clicked button, or None.
        """
        return self.buttons_down_detection(self.buttons, mouse_pos)

    def popupbutton_down_detection(self, mouse_pos):
        """
        Detects if the popup close button was clicked.

        Args:
            mouse_pos (tuple): Mouse position (x, y).

        Returns:
            bool: True if close button clicked, False otherwise.
        """
        if self.popup[3].collidepoint(mouse_pos):  # Check collision with close button rect
            return True

    def menubutton_over_detection(self, mouse_pos):
        """
        Updates hover states for menu buttons based on mouse position.

        Args:
            mouse_pos (tuple): Mouse position (x, y).
        """
        self.buttons_over_detection(self.buttons, mouse_pos)
