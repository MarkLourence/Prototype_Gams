import pygame


class BaseScreen:
    """
    Base class for all screen classes in the GAMS application.
    Provides common functionality for scaling UI elements based on window size
    and helper methods for button rendering and mouse interaction detection.
    This allows derived screens (e.g., Menu, Game) to inherit scaling and button logic
    without duplicating code, promoting DRY principles and easier maintenance.
    """
    def __init__(self, win_size):
        """
        Initializes the BaseScreen with the current window size.
        Sets up the scaling factor for UI elements to ensure consistent appearance
        across different window resolutions.

        Args:
            win_size (tuple): Current window dimensions (width, height) in pixels.
        """
        self.win_size = win_size  # Store the window size for reference in derived classes
        self.init_scale(win_size)  # Calculate and set the scaling factor

    def init_scale(self, win_size):
        """
        Calculates a uniform scaling factor based on the base resolution (640x360).
        This ensures UI elements scale proportionally when the window is resized,
        maintaining aspect ratio and preventing distortion.

        The scale is the minimum of width and height ratios to avoid stretching.

        Args:
            win_size (tuple): Current window dimensions (width, height).
        """
        base_width, base_height = 640, 360  # Base resolution for scaling calculations
        scale_w = win_size[0] / base_width  # Width scaling ratio
        scale_h = win_size[1] / base_height  # Height scaling ratio
        self.scale = min(scale_w, scale_h)  # Use the smaller ratio to maintain proportions

    # Helpers kept intentionally minimal to avoid behavior changes
    def draw_button(self, win, button_data, border_width=2):
        """
        Renders a button on the screen with visual states (normal, hovered, disabled).
        Button data is a tuple containing hover state, enabled state, text surface,
        text position, and button rectangle.

        Args:
            win (pygame.Surface): The Pygame window surface to draw on.
            button_data (tuple): (is_hovered, is_open, text_surface, text_pos, btn_rect)
            border_width (int): Thickness of the button border (default 2, scaled).
        """
        is_hovered, is_open, text, text_pos, btn_rect = button_data  # Unpack button data
        # Determine fill color based on state: gray for hover+enabled, light gray for enabled, dark gray for disabled
        fill = (192, 192, 192) if is_hovered and is_open else \
               (240, 240, 240) if is_open else (132, 132, 132)
        pygame.draw.rect(win, fill, btn_rect)  # Draw the button background
        # Draw the border, ensuring minimum thickness of 1 pixel, scaled by self.scale
        pygame.draw.rect(win, (0, 0, 0), btn_rect, max(1, round(border_width * self.scale)))
        win.blit(text, text_pos)  # Render the button text at the specified position

    def buttons_over_detection(self, buttons, mouse_pos):
        """
        Updates the hover state for all buttons based on mouse position.
        Iterates through the buttons dictionary and sets the hover flag (index 0)
        if the mouse is over the button's rectangle.

        Args:
            buttons (dict): Dictionary of button data, where each value is a list/tuple with hover state at index 0.
            mouse_pos (tuple): Current mouse position (x, y).
        """
        for button in buttons.values():  # Loop through each button's data
            button[0] = button[4].collidepoint(mouse_pos)  # Set hover state if mouse collides with button rect (index 4)

    def buttons_down_detection(self, buttons, mouse_pos):
        """
        Detects which button was clicked (mouse down) at the given position.
        Checks enabled buttons for collision with mouse position and returns the label
        of the first matching button, or None if no button was clicked.

        Args:
            buttons (dict): Dictionary of buttons with labels as keys.
            mouse_pos (tuple): Mouse click position (x, y).

        Returns:
            str or None: Label of the clicked button, or None if no click detected.
        """
        for label, (_, is_open, *_, btn_rect) in buttons.items():  # Unpack button data, skipping unused elements with *_
            if is_open and btn_rect.collidepoint(mouse_pos):  # Check if button is enabled and mouse is over it
                return label  # Return the button label on click
        return None  # No button clicked


