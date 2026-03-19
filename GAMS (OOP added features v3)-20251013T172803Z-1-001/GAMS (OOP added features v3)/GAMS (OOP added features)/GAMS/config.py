# config.py - Configuration file containing all application constants and settings

# Model & Detection Parameters - These control how gesture recognition works
MODEL_PATH_NAME = "model.h5"  # Filename of the trained machine learning model used for gesture prediction
SEQUENCE_LENGTH = 90  # Number of consecutive frames required for a complete gesture sequence input to the model
THRESHOLD = 0.4  # Minimum confidence score required for a gesture prediction to be considered valid (0.0 to 1.0)
ACCEPT_N = 5  # Number of consecutive frames where the same gesture must be detected before accepting it as a valid detection

# Scoring Parameters - These control how performance is evaluated in training mode
PENALTY_RATE = 5.0   # Percentage points deducted per second of delay in completing actions (affects scoring negatively)
TMAX = 20.0          # Maximum time in seconds allowed per action before full penalty is applied (caps the penalty)

# Action Lists - Define the gestures/actions the system can recognize and train on
# Core actions for model prediction (7 actions) - These are the basic gestures the ML model was trained to recognize
ACTIONS = [
    "chocks_inserted", "cut_engine", "start_engine", "stop",  # Aircraft ground operations
    "straight_ahead", "turn_left", "turn_right"  # Directional marshalling signals
]

# Extended actions for training sequence (9 actions) - Includes additional actions for the complete training workflow
TRAINING_ACTIONS = [
    "start_engine", "straight_ahead", "turn_left", "turn_right",  # Basic marshalling sequence
    "stop", "set_brakes", "chocks_inserted", "cut_engine", "all_clear"  # Complete shutdown sequence
]

# Hardware Communication - Settings for communicating with external hardware (ESP8266 microcontroller)
ESP8266_IP = "192.168.4.1"  # IP address of the ESP8266 WiFi module used for hardware control
BASE_URL = f"http://{ESP8266_IP}"  # Base URL constructed from the ESP8266 IP for HTTP requests to control hardware

# Window settings - Configuration for the application window display
WIN_SIZE = (640, 360)  # Default window dimensions in pixels (width, height) for the Pygame application
WIN_CAPTION = "Ground Aircraft Marshalling Simulator"  # Title text displayed in the window title bar
