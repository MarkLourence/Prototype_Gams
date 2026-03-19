
import pygame
import cv2
import mediapipe as mp
import numpy as np
from keras._tf_keras.keras.models import load_model
import time
import sys
import os

from utils import resource_path, pretty_label, command_converter, get_pygame_window_pos, get_score, save_scores_to_csv
from config import SEQUENCE_LENGTH, THRESHOLD, PENALTY_RATE, TMAX, ACCEPT_N


class GameOver:
    """
    Screen class for displaying the game over results after completing a training/assessment session.
    Shows individual action scores, overall score, status, and provides options to exit or retake the mission.
    Handles scoring calculation, CSV saving, and user interaction for navigation.
    """
    def __init__(self, win_size, signal_scores, state):
        """
        Initializes the GameOver screen with the current window size, scores from the session,
        and the game state (including user identity for saving scores).

        Args:
            win_size (tuple): Current window dimensions (width, height).
            signal_scores (dict): Dictionary of scores for each action (e.g., {"Start Engine": 85.5}).
            state (GameState): Shared game state object containing user name/section.
        """
        self.signal_scores = signal_scores  # Store the scores for each marshalling action
        self.state = state  # Reference to game state for user identity
        self.init(win_size)  # Initialize UI elements and prepare content

    def init(self, win_size):
        """
        Sets up the GameOver screen by initializing scaling, fonts, colors,
        and preparing text, buttons, and layout.

        Args:
            win_size (tuple): Window size for scaling calculations.
        """
        self.init_scale(win_size)  # Calculate scaling factor for UI elements
        self.win_size = win_size  # Store window size for layout calculations

        # Initialize fonts for regular text and titles, scaled for resolution
        self.font = pygame.font.SysFont("Franklin Gothic Medium Condensed", int(26 * self.scale))
        self.title_font = pygame.font.SysFont("Franklin Gothic Medium Condensed", int(34 * self.scale), bold=True)

        # Set background and border colors for the popup
        self.bg_color = (255, 255, 255)  # White background
        self.border_color = (0, 0, 0)  # Black borders

        # Prepare all text surfaces (scores, overall, status)
        self.prepare_text()
        # Measure button sizes for consistent layout
        self.measure_button_dimensions()
        # Calculate the popup rectangle dimensions
        self.calculate_popup_rect()
        # Create button objects with positions
        self.prepare_buttons()

    def init_scale(self, win_size):
        """
        Calculates the scaling factor based on base resolution (640x360) to ensure
        UI elements scale proportionally on different window sizes.

        Args:
            win_size (tuple): Current window dimensions.
        """
        base_width, base_height = 640, 360  # Base resolution for consistent scaling
        scale_w = win_size[0] / base_width  # Width ratio
        scale_h = win_size[1] / base_height  # Height ratio
        self.scale = min(scale_w, scale_h)  # Use minimum to maintain aspect ratio

    def prepare_text(self):
        """
        Prepares all text surfaces for display: individual action scores, overall score, and status.
        Calculates the overall percentage and status, then saves scores to CSV with user identity.
        """
        self.text_surfaces = []  # List to hold rendered text surfaces for each action score

        # Ordered list of action labels to ensure consistent display order
        ordered_labels = [
            "Start Engine", "Straight Ahead", "Turn Left", "Turn Right", "Stop", "Set Brakes", "Chocks Inserted", "Cut Engines", "All Clear"
        ]
        total_score = 0  # Accumulate total score for overall calculation
        count = 0  # Count of actions for averaging

        # Render rows in consistent order
        formatted_scores = {}  # Dictionary for CSV saving
        for label in ordered_labels:
            score = float(self.signal_scores[label])  # Get score from input dict
            total_score += score  # Add to total
            count += 1  # Increment count

            formatted_score = int(round(score))  # Round to nearest integer for display
            formatted_scores[label] = formatted_score  # Store for CSV

            text = f"{label}: {formatted_score}%"  # Format text as "Action: Score%"
            surface = self.font.render(text, True, (0, 0, 0))  # Render with black color
            self.text_surfaces.append(surface)  # Add to list

        # Calculate overall score and status using utility function
        overall_pct, status, color = get_score(count, total_score)
        # Retrieve user identity from state if available
        name = getattr(self.state, 'user_name', None) if self.state else None
        section = getattr(self.state, 'user_section', None) if self.state else None
        # Save scores to CSV with identity
        save_scores_to_csv(formatted_scores, overall_pct, status, name, section)

        self.overall_pct = overall_pct  # Store overall percentage
        # Render overall score text
        self.overall_surface = self.title_font.render(
            f"OVERALL SCORE: {self.overall_pct}%", True, (0, 0, 0)
        )
        # Render status text with color based on performance
        self.status_surface = self.font.render(
            f"Status: {status}", True, color
        )

    def prepare_buttons(self):
        """
        Creates button objects for "Exit" and "Retake Mission" with calculated positions
        within the popup rectangle. Buttons are centered horizontally at the bottom.
        """
        self.buttons = {}  # Dictionary to store button data

        # Position buttons at the bottom of the popup with padding
        y = self.popup_rect.bottom - self.button_height - int(15 * self.scale)
        total_button_width = 2 * self.button_width + self.button_spacing  # Total width for two buttons + spacing

        # Center the buttons horizontally
        x_left = self.popup_rect.centerx - (total_button_width // 2)
        x_right = x_left + self.button_width + self.button_spacing

        # Create button data for each button
        for (label, surf, rect), x in zip(self.button_surfaces, [x_left, x_right]):
            btn_rect = pygame.Rect(x, y, self.button_width, self.button_height)  # Button rectangle
            text_pos = (x + (self.button_width - rect.width) // 2, y + (self.button_height - rect.height) // 2)  # Center text
            self.buttons[label] = [False, True, surf, text_pos, btn_rect]  # [hover, enabled, surface, pos, rect]

    def measure_button_dimensions(self):
        """
        Calculates button dimensions based on text size and padding.
        Ensures buttons are uniform in size for a clean layout.
        """
        font_size = int(16 * self.scale)  # Scaled font size
        font = pygame.font.SysFont("Franklin Gothic Medium Condensed", font_size)
        labels = ["Exit", "Retake Mission"]  # Button labels
        surfaces = [font.render(label, True, (0, 0, 0)) for label in labels]  # Render text surfaces

        padding_w = int(25 * self.scale)  # Horizontal padding
        padding_h = int(20 * self.scale)  # Vertical padding
        rects = [s.get_rect() for s in surfaces]  # Get text rectangles

        # Calculate uniform button size based on largest text
        self.button_width = max(r.width for r in rects) + padding_w
        self.button_height = max(r.height for r in rects) + padding_h
        self.button_spacing = int(20 * self.scale)  # Spacing between buttons
        self.button_surfaces = list(zip(labels, surfaces, rects))  # Store for later use

    def calculate_popup_rect(self):
        """
        Calculates the popup rectangle dimensions. The popup takes half the window width
        and full height, positioned at the left side of the screen.
        """
        self.height = int(self.win_size[1])  # Full window height
        self.width = int(self.win_size[0] // 2)  # Half window width

        # Create rectangle starting from top-left
        self.popup_rect = pygame.Rect(
            0,  # x position
            0,  # y position
            self.width,  # width
            self.height  # height
        )

    def draw(self, win):
        """
        Draws the game over popup on the window, including background, text, and buttons.

        Args:
            win (pygame.Surface): The Pygame window surface to draw on.
        """
        pygame.draw.rect(win, self.bg_color, self.popup_rect)  # Draw white background

        spacing = int(28 * self.scale)  # Vertical spacing between text lines
        extra_spacing = int(2 * self.scale)  # Extra space before overall score

        cursor_y = self.popup_rect.y + int(10 * self.scale)  # Start position for text

        # Draw each action score text, centered
        for surf in self.text_surfaces:
            x = self.popup_rect.centerx - surf.get_width() // 2
            win.blit(surf, (x, cursor_y))
            cursor_y += spacing

        cursor_y += extra_spacing  # Add extra space
        # Draw overall score, centered
        x = self.popup_rect.centerx - self.overall_surface.get_width() // 2
        win.blit(self.overall_surface, (x, cursor_y))

        cursor_y += spacing  # Move down for status
        # Draw status text, centered
        x = self.popup_rect.centerx - self.status_surface.get_width() // 2
        win.blit(self.status_surface, (x, cursor_y))

        # Draw buttons with hover effects
        border_width = max(1, round(2 * self.scale))  # Scaled border thickness
        for is_hovered, is_open, text, text_pos, btn_rect in self.buttons.values():
            # Determine fill color based on state
            fill = (192, 192, 192) if is_hovered and is_open else \
                   (240, 240, 240) if is_open else (132, 132, 132)
            pygame.draw.rect(win, fill, btn_rect)  # Draw button background
            pygame.draw.rect(win, (0, 0, 0), btn_rect, border_width)  # Draw border
            win.blit(text, text_pos)  # Draw button text

    def button_over_detection(self, mouse_pos):
        """
        Updates hover states for buttons based on mouse position.

        Args:
            mouse_pos (tuple): Current mouse position (x, y).
        """
        for button in self.buttons.values():
            button[0] = button[4].collidepoint(mouse_pos)  # Set hover if mouse over button rect

    def button_down_detection(self, mouse_pos):
        """
        Detects which button was clicked and returns its label.

        Args:
            mouse_pos (tuple): Mouse click position (x, y).

        Returns:
            str or None: Label of the clicked button ("Exit" or "Retake Mission"), or None.
        """
        for label, (_, is_open, *_, btn_rect) in self.buttons.items():
            if is_open and btn_rect.collidepoint(mouse_pos):  # Check if enabled and clicked
                return label  # Return button label
        return None  # No button clicked
