# Define the GameState class to manage and track the current state of the game/training session
class GameState:
    # Initialize the game state with default values for a new session
    def __init__(self):
        self.training_started = False  # Flag indicating whether training mode has been initiated
        self.assessment_stage = False  # Flag indicating if we're in the final assessment phase
        self.current_action = 0  # Index of the current action in the training sequence (0-based)
        self.signal_detected = False  # Flag set when a gesture signal is detected by the ML model
        self.accepted_for_action = False  # Flag indicating if the current gesture has been accepted for the action
        self.instruction = "None"  # Current instruction text displayed to the user
        self.scores = {}  # Dictionary to store performance scores for each action completed
        # User identity (filled when user provides details) - stores user information for session tracking
        self.user_name = None  # User's name, collected at the start of training
        self.user_section = None  # User's section/department, collected at the start of training

    # Reset the game state to initial values when returning to the menu (clears training progress)
    def reset_for_menu(self):
        self.training_started = False  # Reset training flag to indicate no active training session
        self.assessment_stage = False  # Reset assessment flag since we're back at menu
        self.current_action = 0  # Reset action index to beginning of sequence
        self.signal_detected = False  # Clear any pending signal detection
        self.accepted_for_action = False  # Clear acceptance status
        self.instruction = "None"  # Reset instruction to default "no instruction" state

    # Initialize state for starting a new training session with the first instruction
    def start_training(self, first_instruction):
        self.training_started = True  # Set flag to indicate training has begun
        self.assessment_stage = False  # Ensure we're not in assessment phase yet
        self.current_action = 0  # Start from the first action in the sequence
        self.signal_detected = False  # No signal detected at start
        self.accepted_for_action = False  # No action accepted yet
        self.instruction = first_instruction  # Set the initial instruction to display

    # Advance to the next action in the training sequence after completing the current one
    def advance_action(self, next_instruction):
        self.current_action += 1  # Increment the action index to move to the next action
        self.signal_detected = False  # Reset signal detection for the new action
        self.accepted_for_action = False  # Reset acceptance status for the new action
        self.instruction = next_instruction  # Update the instruction text for the new action

