# Static Hand Gesture UI Control

A real-time hand gesture recognition system that enables touchless control of your PC using static hand poses. Control media playback, volume, and navigation with simple gestures detected from your webcam.

## What It Does

This project uses computer vision and machine learning to recognize five static hand gestures and map them to system actions:

| Gesture | Name | System Action |
|---------|------|---------------|
| Open palm | **Stop** | Play/Pause media (Space key) |
| Closed fist | **Fist** | Toggle mute/unmute |
| Victory sign | **Victory** | Next track/slide (Ctrl+Right) |
| Thumbs up | **Like** | Volume up (+10%) |
| OK sign | **OK** | Volume down (-10%) |

The system runs in real-time (~30 FPS) and provides visual feedback with confidence scores and status indicators.

## How It Works

### Architecture Overview

The system follows a three-stage pipeline:

```
Webcam → MediaPipe Hands → Feature Extraction → Classifier → System Actions
```

1. **Hand Detection & Landmark Extraction** (MediaPipe Hands)
   - Detects hand presence in each video frame
   - Extracts 21 3D landmarks (x, y, z coordinates) per hand
   - Produces a 63-dimensional feature vector (21 × 3)

2. **Gesture Classification** (SVM with RBF kernel)
   - Trained on user-collected gesture data
   - Uses StandardScaler for feature normalization
   - Outputs class prediction (0-4) and confidence probability

3. **Action Triggering** (System Control)
   - Applies confidence threshold (default: 0.75) to prevent false triggers
   - Implements debounce (0.8s) to avoid repeated actions
   - Executes system actions via `pyautogui` (keyboard) and `pycaw` (Windows audio)

### Technical Details

**Feature Extraction:**
- MediaPipe Hands provides normalized 3D coordinates (0-1 range) for 21 hand landmarks
- Features are extracted in real-time from each video frame
- The 63-dim vector represents the spatial configuration of the hand

**Machine Learning Model:**
- **Algorithm:** Support Vector Machine (SVM) with RBF kernel
- **Preprocessing:** StandardScaler (zero mean, unit variance)
- **Hyperparameters:** C=10, gamma='scale'
- **Expected Accuracy:** 97-99.5% on test set with ~200 samples per class
- **Training:** 80/20 stratified train/test split

**Real-time Processing:**
- Processes each frame independently (no temporal modeling)
- Confidence threshold filters low-quality predictions
- Debounce window prevents rapid repeated triggers
- Visual feedback shows prediction, confidence, and trigger status

## Installation

### Prerequisites

- **Windows 10/11** (native, not WSL) - Required for `pycaw` audio control
- **Python 3.10** (3.10.x) - **Important:** MediaPipe does not support Python 3.13+ on Windows
- **Webcam** - For gesture capture
- **Administrator privileges** - May be needed for camera permissions

### Step-by-Step Installation

#### 1. Install Python 3.10

If you don't have Python 3.10:

1. Download Python 3.10 (64-bit) from [python.org](https://www.python.org/downloads/)
2. During installation, **check "Add Python to PATH"**
3. Verify installation:
   ```powershell
   py -3.10 --version
   ```

#### 2. Clone or Download This Repository

```powershell
cd C:\path\to\your\projects
git clone <repository-url>
cd gesture-recog
```

#### 3. Create Virtual Environment

Open PowerShell in the project directory:

```powershell
# Create venv using Python 3.10
py -3.10 -m venv .venv310

# If py -3.10 doesn't work, use full path:
# "C:\Users\<YourName>\AppData\Local\Programs\Python\Python310\python.exe" -m venv .venv310
```

#### 4. Activate Virtual Environment

```powershell
# Allow script execution (one-time per PowerShell session)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Activate the virtual environment
.\.venv310\Scripts\Activate.ps1

# Verify activation (should show (.venv310) in prompt)
python --version  # Should show Python 3.10.x
```

#### 5. Install Dependencies

With the virtual environment activated:

```powershell
# Upgrade pip first
python -m pip install --upgrade pip

# Install all required packages
pip install mediapipe opencv-python scikit-learn joblib numpy pandas matplotlib seaborn pyautogui pycaw comtypes notebook ipykernel
```

**Note:** Installation may take 5-10 minutes. MediaPipe and OpenCV are large packages.

#### 6. Register Jupyter Kernel (Optional but Recommended)

To use the notebook with the correct Python environment:

```powershell
python -m ipykernel install --user --name venv310 --display-name "Python 3.10 (venv310)"
```

#### 7. Configure Camera Permissions

1. Open Windows Settings → Privacy & Security → Camera
2. Enable "Camera access for this device"
3. Enable "Let apps access your camera"
4. Enable "Let desktop apps access your camera"

## Usage

### Quick Start

1. **Start Jupyter Notebook:**

   ```powershell
   # Make sure .venv310 is activated
   python -m notebook
   ```

2. **Open the notebook:**
   - Navigate to `mala_tofu.ipynb` in the browser
   - If prompted, select kernel: **Python 3.10 (venv310)**

3. **Follow the workflow:**
   - **Day 1:** Collect gesture data
   - **Day 2:** Train the classifier
   - **Day 3:** Run real-time control

### Day 1: Data Collection

**Purpose:** Collect training data for the 5 gestures.

**Steps:**

1. Run the **Day 1 data collection cell** in the notebook
2. A video window will open showing your webcam feed
3. For each gesture (0-4):
   - Press the corresponding number key (0, 1, 2, 3, or 4)
   - Hold the gesture pose in front of the camera
   - Keep holding for a few seconds to collect multiple samples
   - Vary your distance, hand rotation, and lighting slightly
4. Aim for **~200+ samples per gesture** (total ~1,000 samples)
5. Press **'q'** in the video window to stop and save

**Output:** `gesture_data.csv` (~100-200 KB)

**Tips:**
- Ensure only one hand is visible
- Keep hand fully in frame
- Vary conditions to improve model robustness
- Watch the sample counter on screen

### Day 2: Training & Evaluation

**Purpose:** Train the SVM classifier and evaluate performance.

**Steps:**

1. Ensure `gesture_data.csv` exists from Day 1
2. Run the **Day 2 training cell** in the notebook
3. The script will:
   - Load the CSV data
   - Split into train/test sets (80/20)
   - Train SVM with StandardScaler
   - Print accuracy and classification report
   - Display confusion matrix
   - Save model as `gesture_classifier.pkl`

**Expected Output:**
- Test accuracy: 97-99.5%
- Classification report with per-class metrics
- Confusion matrix visualization

**Output:** `gesture_classifier.pkl` (trained model)

### Day 3: Real-time Control

**Purpose:** Run live gesture recognition and system control.

**Steps:**

1. Ensure `gesture_classifier.pkl` exists from Day 2
2. Run the **Day 3 real-time control cell** in the notebook
3. A video window will open with:
   - **Green [READY]:** Action will trigger
   - **Orange [WAIT]:** Waiting for debounce cooldown
   - **Red [LOW CONF]:** Confidence too low (adjust threshold)
4. Make gestures in front of the camera
5. Watch console output for action confirmations
6. Press **'q'** in the video window to quit

**Status Indicators:**
- Prediction name and confidence score
- Color-coded trigger status
- Debug messages explaining why actions don't fire

**Troubleshooting:**
- If actions don't trigger, check confidence threshold (default: 0.75)
- Lower threshold for easier triggering: change `confidence_threshold = 0.75` to `0.65`
- Raise threshold to reduce false triggers: change to `0.85` or `0.90`
- Ensure gestures match training data poses

## Functions & Components

### Data Collection Module

**Function:** Captures labeled gesture samples from webcam

**Key Components:**
- `cv2.VideoCapture(0)`: Opens webcam stream
- `mp.solutions.hands.Hands()`: MediaPipe hand detection
- `hands.process(rgb_frame)`: Extracts hand landmarks
- Keyboard input (0-4): Labels current gesture
- CSV writer: Saves 63-dim feature vectors + labels

**Output Format:**
- CSV with 64 columns: `x0, x1, ..., x62, label`
- Each row = one gesture sample
- Label: 0 (Stop), 1 (Fist), 2 (Victory), 3 (Like), 4 (OK)

### Training Module

**Function:** Trains SVM classifier on collected data

**Pipeline:**
1. **Data Loading:** `pd.read_csv('gesture_data.csv')`
2. **Feature/Label Split:** X (63 features), y (labels)
3. **Train/Test Split:** 80/20 stratified (preserves class distribution)
4. **Preprocessing:** `StandardScaler()` normalizes features
5. **Model Training:** `SVC(kernel='rbf', C=10, gamma='scale')`
6. **Evaluation:** Accuracy, classification report, confusion matrix
7. **Model Saving:** `joblib.dump(clf, 'gesture_classifier.pkl')`

**Why SVM?**
- Excellent for small datasets (~1,000 samples)
- Fast inference (~1ms per prediction)
- High accuracy with RBF kernel
- Robust to overfitting with proper regularization

### Real-time Control Module

**Function:** Live gesture recognition and system action execution

**Processing Loop:**
1. **Frame Capture:** `cap.read()` gets webcam frame
2. **Hand Detection:** MediaPipe processes frame
3. **Feature Extraction:** Build 63-dim vector from landmarks
4. **Prediction:** `clf.predict_proba()` gets class + confidence
5. **Trigger Logic:**
   - Check confidence ≥ threshold
   - Check debounce cooldown expired
   - Execute action if both pass
6. **Visual Feedback:** Overlay prediction and status on frame

**System Actions:**
- **Play/Pause:** `pyautogui.press('space')` - Sends Space key
- **Mute Toggle:** `volume.SetMute()` - Windows audio API
- **Next Track:** `pyautogui.hotkey('ctrl', 'right')` - Keyboard shortcut
- **Volume Up/Down:** `volume.SetMasterVolumeLevelScalar()` - Windows audio API

**Safeguards:**
- **Confidence Threshold:** Filters low-quality predictions (default: 0.75)
- **Debounce:** Prevents rapid repeated triggers (0.8s cooldown)
- **Visual Status:** Color-coded feedback (green/orange/red)

## Project Structure

```
gesture-recog/
├── mala_tofu.ipynb          # Main Jupyter notebook (all code)
├── README.md                 # This file
├── docs/
│   ├── task.md              # Day-by-day task checklist
│   ├── idea.md              # Project concept and scope
│   ├── guideline.md         # Course project requirements
│   └── grok-outputs/         # Reference implementations
├── .venv310/                # Python 3.10 virtual environment
├── gesture_data.csv         # Training data (generated Day 1)
└── gesture_classifier.pkl   # Trained model (generated Day 2)
```

## Troubleshooting

### "No module named 'mediapipe'"

**Solution:** 
- Ensure you're using Python 3.10 (not 3.13+)
- Verify virtual environment is activated: `python --version`
- Reinstall: `pip install mediapipe`

### "No module named 'cv2'"

**Solution:**
- Install OpenCV: `pip install opencv-python`
- Restart Jupyter kernel after installation

### Camera not working

**Solution:**
- Check Windows camera permissions (Settings → Privacy → Camera)
- Ensure no other app is using the camera
- Try camera index 1: `cv2.VideoCapture(1)` instead of `0`

### Actions not triggering

**Solution:**
- Lower confidence threshold: `confidence_threshold = 0.65`
- Check console for debug messages
- Ensure gestures match training poses
- Wait for debounce cooldown (0.8s) between triggers

### Volume control not working

**Solution:**
- Ensure running on native Windows (not WSL)
- Check `pycaw` installation: `pip install pycaw`
- Verify audio device is available in Windows

### Jupyter kernel not found

**Solution:**
- Register kernel: `python -m ipykernel install --user --name venv310`
- Restart Jupyter server
- Select kernel: Kernel → Change Kernel → Python 3.10 (venv310)

## Performance

- **Frame Rate:** ~30 FPS on modern hardware
- **Latency:** ~33ms per frame (real-time)
- **Accuracy:** 97-99.5% on test set
- **Model Size:** ~100-500 KB (SVM)
- **Memory:** ~200-500 MB (includes MediaPipe)

## Limitations

- **Windows Only:** Audio control requires Windows APIs (`pycaw`)
- **Single Hand:** Only one hand detected at a time
- **Static Gestures:** No dynamic gesture sequences
- **Lighting Dependent:** Performance degrades in poor lighting
- **Training Required:** Must collect your own gesture data

## Future Improvements

- Add "No gesture" class for better robustness
- Support for left/right hand normalization
- Relative coordinate features (wrist-centered)
- Multi-hand detection
- Dynamic gesture sequences
- Mobile deployment (Android/iOS)

## License

This project is for educational purposes. MediaPipe is licensed under Apache 2.0.

## References

- **MediaPipe Hands:** [Google MediaPipe](https://google.github.io/mediapipe/solutions/hands.html)
- **scikit-learn:** [SVM Documentation](https://scikit-learn.org/stable/modules/svm.html)
- **pycaw:** [Python Core Audio Windows](https://github.com/AndreMiras/pycaw)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review `docs/task.md` for step-by-step guidance
3. Ensure all prerequisites are met (Python 3.10, Windows, etc.)

---

**Happy Gesturing!** 🎭✨
