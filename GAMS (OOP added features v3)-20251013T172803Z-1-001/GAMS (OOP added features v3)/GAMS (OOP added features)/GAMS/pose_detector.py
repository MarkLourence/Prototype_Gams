# Import necessary libraries for computer vision, machine learning, and game development
import pygame  # For creating game surfaces and handling display
import cv2  # OpenCV for camera capture and image processing
import mediapipe as mp  # Google's ML library for pose detection
import numpy as np  # For numerical operations and array handling
from keras._tf_keras.keras.models import load_model  # TensorFlow/Keras for loading the trained gesture recognition model
import sys  # For system operations like exit
import os  # For file path operations

# Import project-specific utilities and configuration
from utils import resources_path  # Function to get the path to resource files
from config import SEQUENCE_LENGTH, THRESHOLD, ACTIONS  # Configuration constants for model parameters

# Define the PoseDetector class responsible for camera input, pose detection, and gesture recognition
class PoseDetector:
    # Set the model path, checking for packaged executable or development environment
    MODEL_PATH = os.path.join(resources_path, "model.h5")  # Default path to the ML model file
    if getattr(sys, 'frozen', False):  # Check if running as a packaged executable (PyInstaller)
        exe_model = os.path.join(os.path.dirname(sys.executable), "model.h5")  # Path for packaged app
        if os.path.exists(exe_model):  # If model exists in executable directory, use that path
            MODEL_PATH = exe_model

    # Initialize the PoseDetector with window size and pygame window reference
    def __init__(self, win_size, win):
        self.win_size = win_size  # Store the window dimensions for scaling calculations
        self.win = win  # Reference to the pygame window surface
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Open the default camera (index 0) with DirectShow backend for Windows
        self.mp_pose = mp.solutions.pose  # Get MediaPipe pose solution
        self.pose = self.mp_pose.Pose()  # Initialize the pose detector model
        self.mp_drawing = mp.solutions.drawing_utils  # Utility for drawing pose landmarks on frames

        # Check if camera opened successfully, exit if not
        if not self.cap.isOpened():
            print("Could not open camera.")  # Error message for camera failure
            pygame.quit()  # Clean up pygame resources
            sys.exit()  # Exit the application

        self.model = load_model(self.MODEL_PATH)  # Load the trained gesture recognition model from file
        self.sequence = []  # List to store sequence of pose keypoints for model input
        self._signal = "NONE"  # Private variable to store the current detected gesture signal
        self._confidence = 0.0  # Private variable to store confidence score of the detection
        self.frame_surface = None  # Pygame surface for the current camera frame
        self.frame_draw_size = None  # Calculated size for drawing the frame in the UI
        self.frame_draw_pos = None  # Calculated position for drawing the frame in the UI
        self.visibility_toggle = "X"  # Toggle for showing/hiding pose landmarks ("X" = show, "+" = hide)

        self.init_opencv(win_size)  # Initialize OpenCV-related settings with current window size

    # Property getter for the current detected signal
    @property
    def signal(self):
        return self._signal  # Return the current gesture signal

    # Property getter for the confidence score
    @property
    def confidence(self):
        return self._confidence  # Return the confidence score of the current detection

    # Initialize OpenCV display settings by reading a dummy frame and calculating scaling
    def init_opencv(self, win_size):
        ret, frame = self.cap.read()  # Read a dummy frame from camera to get dimensions
        if not ret:  # If reading failed, exit the application
            print("Failed to read dummy frame for init.")  # Error message
            pygame.quit()  # Clean up pygame
            sys.exit()  # Exit application

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame from BGR to RGB color space
        frame = np.rot90(frame)  # Rotate frame 90 degrees for proper orientation
        dummy_surface = pygame.surfarray.make_surface(frame)  # Create pygame surface from the frame

        cam_rect = dummy_surface.get_rect()  # Get rectangle dimensions of the camera frame
        max_width = win_size[0] // 2  # Calculate maximum width for camera display (half the window)
        max_height = win_size[1]  # Maximum height is full window height

        scale = min(max_width / cam_rect.width, max_height / cam_rect.height)  # Calculate scaling factor to fit frame
        new_size = (int(cam_rect.width * scale), int(cam_rect.height * scale))  # Calculate new scaled size
        self.frame_draw_size = new_size  # Store the calculated drawing size

        pos_x = win_size[0] - new_size[0] // 2 - max_width // 2  # Calculate X position to center frame in right half
        pos_y = (win_size[1] - new_size[1]) // 2  # Calculate Y position to center frame vertically
        self.frame_draw_pos = (pos_x, pos_y)  # Store the calculated drawing position

    # Main method to update the camera frame and perform gesture recognition
    def update_frame(self):
        ret, frame = self.cap.read()  # Read a new frame from the camera
        if not ret:  # If reading failed, return without processing
            return

        frame = cv2.flip(frame, 1)  # Flip the frame horizontally for mirror effect
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for MediaPipe processing
        
        results = self.pose.process(frame_rgb)  # Process the frame with MediaPipe pose detection
        # Draw pose landmarks on the frame if visibility is enabled
        if results.pose_landmarks and self.visibility_toggle == "X":
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,  # Draw landmarks and connections
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),  # White landmarks
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)  # White connections
            )

        keypoints = self.extract_keypoints_full(results)  # Extract pose keypoints from MediaPipe results
        self.sequence.append(keypoints)  # Add keypoints to the sequence buffer
        if len(self.sequence) > SEQUENCE_LENGTH:  # If sequence is too long, remove oldest frame
            self.sequence.pop(0)

        signal = ""  # Initialize signal variable
        # Check for hardcoded gesture patterns (all_clear and set_brakes)
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark  # Get landmark coordinates
            right_wrist_y = lm[15].y  # Y coordinate of right wrist
            right_elbow_y = lm[14].y  # Y coordinate of right elbow
            left_wrist_y = lm[16].y  # Y coordinate of left wrist
            left_hip_y = lm[23].y  # Y coordinate of left hip
            right_shoulder_y = lm[12].y  # Y coordinate of right shoulder
            right_eye = lm[5].y  # Y coordinate of right eye
            
            # Check conditions for "all_clear" gesture (both arms raised high)
            if (right_wrist_y < right_eye and right_wrist_y < right_elbow_y and
                    left_wrist_y > left_hip_y):
                signal = "all_clear"  # Set signal to all_clear
                self._signal = signal  # Update internal signal
                self._confidence = 1.0  # Set confidence to maximum
            # Check conditions for "set_brakes" gesture (right arm raised, left arm down)
            elif (right_wrist_y < right_shoulder_y and right_wrist_y < right_elbow_y and
                    left_wrist_y > left_hip_y):
                signal = "set_brakes"  # Set signal to set_brakes
                self._signal = signal  # Update internal signal
                self._confidence = 1.0  # Set confidence to maximum

        # If no hardcoded signal detected and sequence is complete, use ML model for prediction
        if signal == "" and len(self.sequence) == SEQUENCE_LENGTH:
            input_seq = np.expand_dims(np.array(self.sequence), axis=0)  # Prepare sequence for model input
            probs = self.model.predict(input_seq, verbose=0)[0]  # Get prediction probabilities from model
            max_idx = np.argmax(probs)  # Find index of highest probability
            self._confidence = float(probs[max_idx])  # Convert confidence to float
            if self._confidence > THRESHOLD:  # If confidence exceeds threshold, accept the prediction
                self._signal = ACTIONS[max_idx]  # Set signal to the predicted action
            else:
                self._signal = "NONE"  # Reset signal if confidence is too low

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert back to RGB for display
        frame_rgb = cv2.flip(frame_rgb, 1)  # Flip again for consistent orientation
        frame_rgb = np.rot90(frame_rgb)  # Rotate for pygame display
        self.frame_surface = pygame.surfarray.make_surface(frame_rgb)  # Create pygame surface from processed frame

    # Extract full pose keypoints from MediaPipe results as a flattened array
    def extract_keypoints_full(self, results):
        if not results.pose_landmarks:  # If no landmarks detected, return zeros
            return np.zeros(33 * 3)  # 33 landmarks * 3 coordinates (x, y, z) = 99 zeros
        return np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()  # Extract and flatten coordinates

    # Release camera resources when done
    def release(self):
        self.cap.release()  # Release the camera capture object
