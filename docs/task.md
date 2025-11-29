# Task — Static Hand Gesture UI Control

## Day 0 — Local Setup on Windows (10–20 min)
- [x] Work on native Windows (not WSL). pycaw is Windows-only.
- [x] Install Python 3.10+ on Windows 10/11
- [x] Create and activate virtual environment (PowerShell):
  - [x] `python -m venv .venv`
  - [x] `.\.venv\Scripts\Activate.ps1` (Allow script: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)
- [x] Install dependencies:
  - [x] `pip install mediapipe opencv-python scikit-learn joblib numpy pandas matplotlib seaborn pyautogui pycaw comtypes`
- [x] Optional: install Jupyter for notebooks (`pip install jupyter`)
- [x] Verify webcam access (Privacy & security → Camera → allow apps/desktop apps)
- [x] Keep this `docs/task.md` open in your editor for checklist tracking

## Day 1 — Data Collection (30–50 min total)
- [ ] Confirm virtualenv is active and packages installed
- [ ] Create a new local Jupyter notebook (or `.py` script) in project root
- [ ] Paste & run the full data collection script from `docs/grok-outputs/grok-1.md` (Step 2)
- [ ] Collect all 5 gestures:
  - [ ] Open palm (Stop) → ~200+ samples
  - [ ] Fist → ~200+ samples
  - [ ] Victory → ~200+ samples
  - [ ] Thumbs Up → ~200+ samples
  - [ ] OK sign → ~200+ samples
- [ ] Vary distance, slight rotation, lighting a bit
- [ ] Save → gesture_data.csv (file size ~100–200 KB)

## Day 2 — Training & Evaluation (20–40 min)
- [ ] Load CSV with pandas
- [ ] Train/test split (stratified)
- [ ] Train SVM pipeline (StandardScaler + SVC RBF C=10)
- [ ] Print classification report + accuracy
- [ ] Plot confusion matrix (seaborn heatmap → Fig 3 for report)
- [ ] Quick ablation: also train MLP and KNN → make comparison table
- [ ] joblib.dump model → gesture_classifier.pkl
- [ ] (Optional) Try relative coordinates version for +1–2% boost

## Day 3 — Real-time Demo + System Control (30–60 min)
- [ ] Run locally on Windows (pycaw volume control is Windows-only)
- [ ] Paste the full real-time script from `docs/grok-outputs/grok-1.md` (Step 4)
- [ ] Test each gesture → make sure actions fire correctly
- [ ] Tune confidence threshold (0.90–0.95) and debounce (~0.8 s) until zero spam
- [ ] Test with Spotify/YouTube open → feels like magic
- [ ] Record 30–45 second demo video (use OBS or just phone pointing at screen)

## Day 4 — Report & Polish (2–3 hours)
- [ ] Ensure notebook is saved as .ipynb (final deliverable)
- [ ] Make figures in notebook:
  - Fig 1: 5 gestures with landmarks (screenshot from data collection)
  - Fig 2: pipeline diagram (you can draw in draw.io or even Canva → 5 min)
  - Table 1: classification report
  - Fig 3: confusion matrix
- [ ] Write 4-page CVPR LaTeX (use Overleaf template)
  - Page 1: Title, abstract, intro, Fig 1
  - Page 2: Related work, method, Fig 2
  - Page 3: Dataset, experiments, Table 1, Fig 3
  - Page 4: Implementation, results, conclusion, references
- [ ] Add QR code to demo video in report (optional but looks pro)

## Stretch / Bonus (if you're ahead)
- [ ] Add "No gesture" class (collect 300 background samples)
- [ ] Relative + normalized coordinates
- [ ] Handedness flip (so left hand works too)
- [ ] Mobile test (run on phone with Colab + ngrok or just local)

## Done when:
- Notebook/scripts run end-to-end locally on Windows
- Demo video shows all 5 gestures controlling actual PC
- Report = exactly 4 pages + references
- You feel like a wizard