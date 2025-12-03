# Static Hand Gesture UI Control

Real-time media control through four static gestures and ergonomic zone-based swipes. MediaPipe Hands provides landmarks, an SVM classifier recognizes static poses, and a zone detector handles track navigation without fast arm motions.

## Capabilities
- Static gestures (Stop, Fist, Like, Thumbs Down) mapped to play/pause, mute, and volume up/down.
- Zone-based navigation splits the frame into LEFT/CENTER/RIGHT bands and triggers next/previous track when the wrist crosses from CENTER to an edge.
- Confidence gating (≥0.88), three-frame stability, and cooldown timers suppress accidental triggers.
- Visual overlays show zones, prediction state, and action confirmations in the live window.

## Gesture Mapping

| Mode | Gesture | Action |
|------|---------|--------|
| Static (CENTER) | Stop / open palm | Play / Pause |
| Static (CENTER) | Fist | Toggle mute |
| Static (CENTER) | Like / thumbs up | Volume +10% |
| Static (CENTER) | Thumbs Down | Volume −10% |
| Navigation | CENTER → LEFT zone | Previous track |
| Navigation | CENTER → RIGHT zone | Next track |

## System Overview

```
Webcam → MediaPipe Hands → 63D features → SVM classifier → System actions
                                 ↘ Wrist x-position → Zone detector ↗
```

- **Hand Landmarks:** MediaPipe Hands (21 points × x,y,z) per hand.
- **Classifier:** scikit-learn SVM (RBF kernel, C=10, gamma='scale') wrapped in a StandardScaler pipeline; probabilities drive the confidence check.
- **Static Logic:** Requires confidence ≥0.88, three consecutive agreeing frames, and 1.1 s cooldown between activations.
- **Zone Logic:** Wrist x-position defines zone transitions. ONLY transitions from CENTER to an edge trigger swipes, with a 1.5 s cooldown. Static recognition pauses while a swipe is cooling down.
- **System Control:** `pyautogui` handles media keys, `pycaw` adjusts Windows audio via COM interfaces.

## Requirements
- Windows 10/11 (native, not WSL) for camera + `pycaw`.
- Python 3.10.x (MediaPipe Windows wheels stop at 3.10 today).
- Webcam with stable lighting.
- Optional: existing `gesture_data.csv` and `gesture_classifier.pkl` included for demo use; delete them if you wish to recollect/train.

## Installation
1. **Clone the repo**
   ```powershell
   cd C:\path\to\projects
   git clone <repository-url>
   cd gesture-recog
   ```
2. **Create venv (Python 3.10)**
   ```powershell
   py -3.10 -m venv .venv310
   ```
3. **Activate + allow scripts (per shell session)**
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv310\Scripts\Activate.ps1
   ```
4. **Install dependencies**
   ```powershell
   python -m pip install --upgrade pip
   pip install mediapipe opencv-python scikit-learn joblib numpy pandas matplotlib seaborn pyautogui pycaw comtypes notebook ipykernel
   ```
5. **Register the kernel (optional but handy)**
   ```powershell
   python -m ipykernel install --user --name venv310 --display-name "Python 3.10 (venv310)"
   ```
6. **Enable camera access** in Windows Privacy & Security settings.

## Notebook Workflow (`gesture_main.ipynb`)

### 1. Environment Check
Cell 2 prints the interpreter path/version; helpful when multiple Python installs coexist.

### 2. Data Collection
- Launches a dual-hand capture UI. Press `0-3` to label Stop, Fist, Like, or Thumbs Down. Each detected hand yields an entry.
- Goal: ≥200 samples per class (≈800 total). Use different distances, rotations, and lighting.
- Output saved to `gesture_data.csv`. A sample file already ships for quick demos.

### 3. Model Training
- Loads `gesture_data.csv`, performs an 80/20 stratified split, scales features, trains the SVM, prints metrics, and renders a confusion matrix.
- Saves the fitted pipeline to `gesture_classifier.pkl` for the real-time loop.

### 4. Real-Time Control
- Opens a 1600×900 preview window with zone markers, stability counters, and action overlays.
- Static gestures only register in the CENTER zone; navigation requires sliding into LEFT/RIGHT with a brief return to CENTER between swipes.
- Console logs reiterate configured thresholds, cooldowns, and any failures to trigger.

## Project Structure
```
gesture-recog/
├── gesture_main.ipynb        # Notebook covering collection → training → demo
├── README.md
├── docs/                  # Course notes and requirements
├── gesture_data.csv       # Sample dataset (safe to delete/regenerate)
└── gesture_classifier.pkl # Sample trained model
```

## Troubleshooting
- **mediapipe / cv2 missing:** Ensure the venv is active (`python --version` should show 3.10.x) and reinstall `pip install mediapipe opencv-python`.
- **Camera not detected:** Check Windows privacy toggles, ensure no other app owns the device, try `cv2.VideoCapture(1)` if you have multiple cameras.
- **Actions never fire:** Confirm the hand sits fully in the CENTER zone, keep the gesture steady for ~1 s, and verify confidence ≥0.88 in the overlay. Reduce `confidence_threshold` to 0.82 if needed.
- **Volume controls fail:** Run natively on Windows and make sure an audio endpoint is active. `pycaw` does not work inside WSL.
- **Kernel mismatch:** Reinstall the ipykernel using the venv interpreter and pick it from Jupyter’s Kernel selector.

## References
- MediaPipe Hands: https://google.github.io/mediapipe/solutions/hands
- scikit-learn SVM: https://scikit-learn.org/stable/modules/svm.html
- pycaw: https://github.com/AndreMiras/pycaw

---
This project is intended for academic use. Contributions or bug reports are welcome via pull request or issue.
