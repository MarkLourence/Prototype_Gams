import csv  # Import csv module for reading and writing CSV files, used to store user scores
import os  # Import os module for operating system dependent functionality, such as file path operations
import sys  # Import sys module for system-specific parameters and functions, like checking if running in frozen mode
from datetime import datetime  # Import datetime for getting current date and time, used in score logging
import requests  # Import requests for making HTTP requests, used to send commands to external devices
import pygame  # Import pygame for game development, here used for accessing window information
import ctypes  # Import ctypes for calling C functions, used to get window position on Windows

from config import BASE_URL  # Import BASE_URL from config module, which is the base URL for API commands

# Define a function to get the correct path to resources, handling both development and packaged (frozen) environments
def resource_path(relative_path: str) -> str:
    # Check if the script is running as a frozen executable (e.g., bundled by PyInstaller)
    if getattr(sys, 'frozen', False):
        # If frozen, try to get the temporary folder where PyInstaller extracts bundled files
        base = getattr(sys, '_MEIPASS', None)
        if base:
            # Return the path joined with the relative path in the temp folder
            return os.path.join(base, relative_path)
        # Fallback: use the directory of the executable file
        return os.path.join(os.path.dirname(sys.executable), relative_path)
    else:
        # In development mode, return path relative to the current script's directory
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)

# Set a module-level variable for the resources path, used throughout the application for asset locations
resources_path = resource_path("resources")

# Define a function to get the directory for user data, ensuring persistence across runs
def get_user_data_dir(app_name="GroundAircraftMarshalling"):
    # If running as a frozen executable, use the directory of the executable for data storage
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        # In development, use the directory of the current script
        return os.path.dirname(os.path.abspath(__file__))

# Get the user data directory and create it if it doesn't exist
user_data_dir = get_user_data_dir()
os.makedirs(user_data_dir, exist_ok=True)  # Create the directory, ignoring if it already exists

# Define a function to save scores to a CSV file in the user data directory
def save_scores_to_csv(scores, overall_score, status, name=None, section=None):
    # Construct the full path to the scores.csv file in the user data directory
    filename = os.path.join(user_data_dir, "scores.csv")
    # Check if the file already exists to determine if we need to write headers
    file_exists = os.path.isfile(filename)
    # Get the current date and time as a formatted string
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Prepare the data row: timestamp, name, section, individual scores, overall score, status
    row = [now, name or "", section or ""] + [scores.get(key, "") for key in scores.keys()] + [overall_score] + [status]
    # Open the file in append mode with newline handling for CSV
    with open(filename, mode="a", newline="") as f:
        writer = csv.writer(f)
        # If the file doesn't exist, write the header row first
        if not file_exists:
            header = ["Date & Time", "Name", "Section"] + list(scores.keys()) + ["Overall Score"] + ["Status"]
            writer.writerow(header)
        # Write the data row to the file
        writer.writerow(row)

# Define a function to calculate the overall score percentage and determine status and color
def get_score(count, total_score):
    # Avoid division by zero if no actions were performed
    if count == 0:
        overall_pct = 0
    else:
        # Calculate the percentage by dividing total score by count and rounding
        overall_pct = int(round(total_score / count, 0))
    # Determine status and color based on percentage thresholds
    if overall_pct >= 90:
        status = "EXCELLENT"
        color = (0, 128, 0)  # Green color for excellent
    elif overall_pct >= 75:
        status = "GOOD"
        color = (0, 128, 0)  # Green color for good
    elif overall_pct >= 50:
        status = "NEEDS IMPROVEMENT"
        color = (200, 140, 0)  # Orange color for needs improvement
    else:
        status = "UNSATISFACTORY"
        color = (200, 0, 0)  # Red color for unsatisfactory
    # Return the percentage, status, and color
    return overall_pct, status, color

# Define a function to get the position of the Pygame window on the screen
def get_pygame_window_pos():
    # Get the window handle from Pygame's display info
    hwnd = pygame.display.get_wm_info()['window']
    # Create a RECT structure to hold window coordinates
    rect = ctypes.wintypes.RECT()
    # Call Windows API to get the window rectangle
    ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
    # Return the top-left coordinates of the window
    return (rect.left, rect.top)

# Define a function to convert internal action keys to pretty, human-readable labels
def pretty_label(key):
    # Dictionary mapping internal keys to display labels
    mapping = {
        "start_engine": "Start Engine",
        "straight_ahead": "Straight Ahead",
        "turn_left": "Turn Left",
        "turn_right": "Turn Right",
        "stop": "Stop",
        "set_brakes": "Set Brakes",
        "chocks_inserted": "Chocks Inserted",
        "cut_engine": "Cut Engines",
        "all_clear": "All Clear",
    }
    # Return the mapped label for the given key
    return mapping[key]

# Define a function to convert action labels to command strings for external communication
def command_converter(label):
    # Dictionary mapping action labels to command strings
    mapping = {
        "start_engine": "engine_on",
        "straight_ahead": "forward",
        "turn_left": "left",
        "turn_right": "right",
        "stop": "stop",
        "cut_engine": "engine_off",
    }
    # Try to return the mapped command, return None if key not found
    try:
        return mapping[label]
    except KeyError:
        return

# Define a function to send a command to an external device via HTTP request
def send_command(command, timeout=1.0):
    # Construct the full URL by appending the command to the base URL
    url = f"{BASE_URL}/{command}"
    # Attempt to send a GET request to the URL with the specified timeout
    try:
        requests.get(url, timeout=timeout)
        # Print success message if request succeeds
        print(f"Sent command: {command}")
    # Catch any request exceptions (e.g., connection errors)
    except requests.exceptions.RequestException as e:
        # Print error message with details
        print(f"Error sending request to {url}: {e}")
        # Provide user hint about network connection
        print("Check that you are connected to the ESP8266's Wi-Fi network.")
