# gesture-recog
Gesture UI Control (static poses) – MediaPipe Hands for keypoints + simple classifier (MLP/SVM) for 4–5 gestures.

## Quickstart (Local Windows)

Run natively on Windows 10/11 (not WSL). Requires a webcam.

1) Create and activate a virtual environment (PowerShell):

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

2) Install dependencies:

```
pip install mediapipe opencv-python scikit-learn joblib numpy pandas matplotlib seaborn pyautogui pycaw comtypes
```

3) (Optional) Install Jupyter Notebook into the same environment:

```
pip install jupyter
```

4) Camera permission (Windows): Settings → Privacy & security → Camera → allow apps and desktop apps.

5) Follow `docs/task.md` for the full local workflow:
- Day 1: Data collection (save `gesture_data.csv`)
- Day 2: Train SVM, save `gesture_classifier.pkl`
- Day 3: Real-time control (pyautogui + pycaw)
- Day 4: Report & polish

Reference details and ready-to-paste code are in `docs/grok-outputs/grok-1.md`.