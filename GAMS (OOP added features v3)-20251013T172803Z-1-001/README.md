# GAMS: Ground Aircraft Marshalling Simulator

🧩 **Project Overview**

GAMS is an interactive simulator for training and assessing ground aircraft marshalling skills using machine learning-powered pose detection. The system integrates a trained neural network model to recognize hand signals in real-time from a webcam feed, combined with a Pygame-based interface for visual guidance, audio feedback, and performance evaluation. Its purpose is to provide an automated, cost-effective training tool for aviation personnel, simulating marshalling procedures through ML-driven prediction and interactive visualization. The ML model controls game behavior by detecting and validating user gestures, while Pygame renders the simulation environment, displaying predictions, instructions, and progress.

⚙️ **Tech Stack**

- **Languages**: Python 3.x
- **Frameworks/Libraries**:
  - TensorFlow/Keras (for ML model training and inference)
  - Pygame (for GUI, rendering, and event handling)
  - OpenCV (for video capture and processing)
  - MediaPipe (for pose estimation and landmark detection)
  - NumPy (for numerical computations and data handling)
- **Hardware Dependencies**: Webcam (for pose detection), optional ESP8266 microcontroller (for hardware control via Wi-Fi)

📂 **Project Structure**

```
GAMS (OOP added features v3)/
├── GAMS/
│   ├── main.py                 # Entry point, initializes AppController
│   ├── app_controller.py       # Main application controller, manages game loops and states
│   ├── config.py               # Configuration constants (model paths, actions, scoring, etc.)
│   ├── utils.py                # Utility functions (resource paths, CSV saving, command sending)
│   ├── game_state.py           # GameState class for managing shared state
│   ├── pose_detector.py        # PoseDetector class for ML inference and camera handling
│   ├── ui_manager.py           # UIManager class for Pygame UI elements and rendering
│   ├── audio_manager.py        # AudioManager class for loading and playing audio files
│   ├── video_manager.py        # VideoManager class for playing introduction and event videos
│   ├── screens/
│   │   ├── __init__.py
│   │   ├── base_screen.py      # Base screen class (if extended)
│   │   ├── menu.py             # Menu screen for navigation
│   │   ├── game.py             # Training & Assessment screen with ML integration
│   │   ├── game_over.py        # Game Over screen with results
│   │   ├── realtime.py         # Real-Time prediction screen
│   │   └── __pycache__/        # Python bytecode cache
│   ├── resources/
│   │   ├── background.png      # UI background image
│   │   ├── icon.ico/.png       # Application icons
│   │   ├── model.h5            # Trained Keras ML model
│   │   ├── warning_audio.mp3   # Warning sound effect
│   │   ├── bookends/           # Introduction and ending audio/video
│   │   ├── detection_audio/    # Audio feedback for detected actions
│   │   ├── guide_videos/       # Frame-based guide animations for actions
│   │   └── instruction_audio/  # Instructional audio prompts
│   ├── scores.csv              # CSV file for storing user scores
│   └── __pycache__/            # Python bytecode cache
```

The root folder contains the main project code, with ML-related modules (pose_detector.py) and game-related modules (screens/, ui_manager.py) separated. Resources include the trained model, audio, video, and UI assets.

🧠 **Machine Learning Component**

The ML component uses a neural network (likely a combination of CNN and RNN layers) for sequence-based pose detection. The model processes sequences of 90 frames of pose keypoints extracted from MediaPipe's holistic pose estimation. Training data consists of labeled sequences of hand/arm gestures corresponding to marshalling actions (e.g., "chocks_inserted", "start_engine"). Preprocessing involves normalizing keypoints and handling missing data. The model.h5 file stores the trained weights and architecture. Retraining involves collecting new sequences, preprocessing, and using Keras' fit method. Evaluation can be done via accuracy metrics on a validation set. Updates require saving the new model.h5 and reloading in the application.

🎮 **Pygame Visualization Component**

Pygame handles rendering of the camera feed, UI panels, prediction text, progress bars, and guide animations. The system features multiple game states: menu, training/assessment, game over, and real-time mode. Event loops manage user input, resizing, and button interactions. The ML model interacts with the game by providing real-time predictions that drive scoring, audio feedback, and progression through training sequences. For example, correct gesture detection advances the action and triggers videos/audio.

🚀 **Installation & Setup**

1. **Clone the Repository**:
   ```
   git clone <repository-url>
   cd GAMS
   ```

2. **Create Virtual Environment** (recommended):
   ```
   python -m venv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On Linux/Mac
   ```

3. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```
   (Note: requirements.txt should include tensorflow, pygame, opencv-python, mediapipe, numpy, requests)

4. **Prepare Resources**:
   - Ensure model.h5 is in resources/
   - Verify audio/video files are present in respective subfolders
   - Connect webcam for pose detection

5. **Run the Application**:
   ```
   python main.py
   ```

**OS-Specific Notes**:
- Windows: Ensure Visual C++ Redistributables for OpenCV.
- Linux/Mac: Install system dependencies for OpenCV (e.g., `sudo apt install libopencv-dev` on Ubuntu).
- For hardware control, connect to ESP8266 Wi-Fi network (IP: 192.168.4.1).

▶️ **Usage Guide**

- **Run Training Mode**: Select "TRAINING & ASSESSMENT" from menu, enter name/section, follow on-screen instructions and gestures.
- **Run Real-Time Mode**: Select "REAL-TIME" for live prediction display.
- **Controls**: Use mouse for buttons, spacebar to skip audio/videos, ESC to exit.
- **Visualization**: Camera feed shows pose landmarks (toggle visibility), predictions update in real-time, progress bar tracks training advancement.

🧱 **Code Architecture & Design**

The codebase follows OOP principles with encapsulation (e.g., PoseDetector handles ML logic), inheritance (screens inherit from base classes), and polymorphism (different screen classes implement draw/update methods). Components interact via composition: AppController orchestrates screens and managers; Game screen uses PoseDetector for ML, UIManager for UI, AudioManager for sound. Design patterns include Observer (state changes trigger UI updates) and Factory (screen instantiation). Flow: main.py → AppController → Screens (Game/RealTime) → Managers (UI, Audio, Video) → PoseDetector.

🧪 **Testing**

Tests are not explicitly defined in the codebase, but can be added using unittest or pytest. Sample test cases:
- Model evaluation: Load model.h5, run inference on test sequences, assert accuracy > threshold.
- Simulation testing: Mock pose detection, verify UI updates and scoring logic.
Run tests with `python -m pytest tests/` (create tests/ directory with test files).

🧰 **Troubleshooting & Debugging**

| Error | Cause | Fix |
|-------|-------|-----|
| ModuleNotFoundError (e.g., tensorflow) | Missing dependency | Install via `pip install tensorflow` |
| OSError: Unable to open file model.h5 | Model file missing/corrupted | Verify model.h5 in resources/, retrain if needed |
| pygame.error: video system not initialized | Pygame init failure | Ensure display server running, try `export DISPLAY=:0` on Linux |
| Tensor shape mismatch | Input sequence length wrong | Check SEQUENCE_LENGTH in config.py matches model input |
| Frame rate lag in Pygame | High CPU usage | Reduce frame processing in pose_detector.py, optimize MediaPipe settings |
| cv2.VideoCapture(0) fails | Webcam not accessible | Check permissions, try different index (e.g., 1) |

**Debugging Steps**:
- Enable verbose logs: Add `import logging; logging.basicConfig(level=logging.DEBUG)` in main.py.
- Check dataset paths: Print `resources_path` in utils.py.
- Display intermediate predictions: Add print statements in pose_detector.py update_frame().
- Use breakpoints: Run with `python -m pdb main.py`, set breakpoints in suspected areas.

🧑‍💻 **Contributing**

- Fork the repository and create a feature branch.
- Follow PEP8 style; use black for formatting.
- Commit messages: "feat: add new action", "fix: resolve pose detection lag".
- Test changes locally before PR.

📘 **Author & License**

Author: TODO  
License: TODO  

🌟 **Optional Enhancements**

- Badges: ![Build Status](https://img.shields.io/badge/build-passing-brightgreen) ![License](https://img.shields.io/badge/license-MIT-blue)
- Screenshots: Include images of menu, training screen, real-time predictions.
- Changelog: v1.0 - Initial release with ML pose detection and Pygame UI.
