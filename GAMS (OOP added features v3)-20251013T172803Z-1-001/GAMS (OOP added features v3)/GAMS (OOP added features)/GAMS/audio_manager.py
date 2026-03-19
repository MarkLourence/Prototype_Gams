# Import necessary modules: os for file path operations, pygame for audio handling, and resources_path from utils for resource directory access
import os
import pygame
from utils import resources_path


# Define the AudioManager class to handle all audio-related operations in the application
class AudioManager:
    # Initialize the AudioManager with optional audio parameter and set up dictionaries for different audio types
    def __init__(self, audio=None):
        self.audio = audio  # Store optional audio reference (currently unused in this implementation)
        self.detection_audio = {}  # Dictionary to hold detection audio clips keyed by action names
        self.instruction_audio = {}  # Dictionary to hold instruction audio clips keyed by action names
        self.bookends_audio = {}  # Dictionary to hold bookend audio clips (introduction and ending) keyed by name
        self.warning_audio = None  # Single warning audio clip, initialized to None

    # Loading section: Methods to load audio files from disk into memory

    # Load detection audio files for each action; these are played when gestures are detected
    def load_detection_audio(self, actions):
        audio_dir = os.path.join(resources_path, "detection_audio")  # Construct path to detection audio directory
        for action in actions:  # Iterate through each action in the provided list
            path = os.path.join(audio_dir, f"{action}_dec.mp3")  # Build file path for this action's detection audio
            if os.path.exists(path):  # Check if the audio file exists at the path
                try:
                    self.detection_audio[action] = pygame.mixer.Sound(path)  # Load the audio file into pygame Sound object and store in dictionary
                except pygame.error:  # If loading fails (e.g., corrupted file), silently skip without crashing
                    pass

    # Load instruction audio files for each action; these provide verbal instructions for gestures
    def load_instruction_audio(self, actions):
        audio_dir = os.path.join(resources_path, "instruction_audio")  # Construct path to instruction audio directory
        for action in actions:  # Iterate through each action in the provided list
            path = os.path.join(audio_dir, f"{action}_ins.mp3")  # Build file path for this action's instruction audio
            if os.path.exists(path):  # Check if the audio file exists at the path
                try:
                    self.instruction_audio[action] = pygame.mixer.Sound(path)  # Load the audio file into pygame Sound object and store in dictionary
                except pygame.error:  # If loading fails, silently skip
                    pass

    # Load the warning audio file; this is a single file played for warnings
    def load_warning(self):
        path = os.path.join(resources_path, "warning_audio.mp3")  # Construct path to warning audio file
        if os.path.exists(path):  # Check if the file exists
            try:
                self.warning_audio = pygame.mixer.Sound(path)  # Load the audio file into pygame Sound object
            except pygame.error:  # If loading fails, set to None to indicate unavailability
                self.warning_audio = None

    # Load bookend audio files (introduction and ending); these are played at the start and end of sessions
    def load_bookends(self):
        audio_dir = os.path.join(resources_path, "bookends")  # Construct path to bookends audio directory
        for name in ["introduction", "ending"]:  # Iterate through the two bookend types
            path = os.path.join(audio_dir, f"{name}.mp3")  # Build file path for this bookend audio
            if os.path.exists(path):  # Check if the file exists
                try:
                    self.bookends_audio[name] = pygame.mixer.Sound(path)  # Load the audio file and store in dictionary
                except pygame.error:  # If loading fails, silently skip
                    pass

    # Controls section: Methods to control audio playback

    # Stop all currently playing audio channels
    def stop(self):
        pygame.mixer.stop()  # Call pygame's mixer stop to halt all audio playback

    # Play detection audio for a specific action; stops any current audio first
    def play_detection(self, action):
        snd = self.detection_audio.get(action)  # Retrieve the Sound object for the given action from the dictionary
        if snd:  # If the sound exists (was loaded successfully)
            pygame.mixer.stop()  # Stop any currently playing audio
            snd.play()  # Play the detection audio for this action

    # Play instruction audio for a specific action; stops any current audio first
    def play_instruction(self, action):
        snd = self.instruction_audio.get(action)  # Retrieve the Sound object for the given action
        if snd:  # If the sound exists
            pygame.mixer.stop()  # Stop any currently playing audio
            snd.play()  # Play the instruction audio

    # Play bookend audio (introduction or ending); stops any current audio first
    def play_bookend(self, name):
        snd = self.bookends_audio.get(name)  # Retrieve the Sound object for the given bookend name
        if snd:  # If the sound exists
            pygame.mixer.stop()  # Stop any currently playing audio
            snd.play()  # Play the bookend audio

    # Play warning audio; stops any current audio first
    def play_warning(self):
        if self.warning_audio:  # If the warning audio was loaded successfully
            pygame.mixer.stop()  # Stop any currently playing audio
            self.warning_audio.play()  # Play the warning audio


