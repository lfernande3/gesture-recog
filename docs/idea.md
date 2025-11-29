## Project Ideas

Gesture UI Control (static poses) – MediaPipe Hands for keypoints + simple classifier (MLP/SVM) for 4–5 gestures.

### Project title
Static Hand Gesture UI Control using MediaPipe Hands + Simple Classifier

### Problem statement
Enable touchless control of common PC functions (media playback, slide navigation, volume) using a small set of easy, static hand poses detected in real time from a webcam feed.

### Scope and gestures
- Single-hand, static poses only (no sequence modeling).
- Five classes (labels 0–4) with distinct landmark patterns:
  - 0: Open palm (“Stop”) → Play/Pause (Space)
  - 1: Closed fist (“Fist”) → Mute/Unmute (system)
  - 2: Victory/Peace (“Victory”) → Next track/slide (Ctrl+Right)
  - 3: Thumbs Up (“Like”) → Volume Up (+10%)
  - 4: OK sign (“OK”) → Volume Down (-10%)

### Technical approach
- Landmark extraction: MediaPipe Hands (21 landmarks × {x,y,z} → 63-dim feature vector).
- Classifier: scikit-learn SVM (RBF) with StandardScaler (baseline), optionally compare MLP and KNN.
- Trigger logic: confidence threshold and debounce window to prevent repeated actions.
- System actions: `pyautogui` for key presses; `pycaw` for Windows volume control.

### Dataset and collection
- Collect ~180–220 samples per gesture (≈ 900–1,100 samples total).
- Procedure: live capture via OpenCV; press keys 0–4 to label while holding the pose; save CSV with 63 features + label.
- Tips: vary distance, rotation, lighting; ensure only one hand visible; mirror camera for intuitive UX.

### Model and training
- Split: 80/20 stratified train/test.
- Pipeline: `StandardScaler` + `SVC(kernel='rbf', C=10, gamma='scale')`.
- Metrics: overall accuracy, per-class precision/recall/F1, confusion matrix.
- Expected performance: 97–99.5% test accuracy with the above data volume.
- Save trained model with `joblib` for real-time inference.

### Real-time application
- Input: webcam (OpenCV), mirrored display, MediaPipe inference each frame.
- Inference: assemble 63-dim vector; run classifier; obtain class + probability.
- UX safeguards:
  - Confidence threshold: p ≥ 0.90 to accept.
  - Debounce: require ~0.7–0.8 s between identical triggers.
- Actions:
  - Space (play/pause), Ctrl+Right (next), toggle mute, ±10% volume steps.

### Evaluation
- Quantitative: accuracy, classification report, confusion matrix on held-out test set.
- Qualitative: real-time latency, perceived stability, false trigger rate in everyday lighting.
- Ablations: with/without scaling; SVM vs MLP vs KNN; optional relative coordinate normalization.

### Deliverables
- One Jupyter/Colab notebook with:
  - Environment setup, data collection UI, training, evaluation, and real-time demo cells.
  - Saved artifacts: `gesture_data.csv`, `gesture_classifier.pkl`.
- Short demo video of real-time control.
- 4-page CVPR-style report with pipeline diagram, dataset summary, metrics table, confusion matrix, and discussion.

### Timeline (suggested)
- Day 1: Set up notebook and data collection; capture all classes.
- Day 2: Train/evaluate models; finalize SVM; save model; start report figures.
- Day 3: Build real-time app; integrate actions; tune confidence/debounce; record demo.
- Day 4: Write and finalize 4-page report; polish results and ablations.

### Risks and mitigations
- False triggers: raise confidence threshold; add debounce; consider a “No gesture” class.
- Left vs right hand differences: optionally normalize handedness (flip landmarks).
- Lighting and scale changes: collect data across varying conditions; use relative coordinates (subtract wrist, scale by palm size) if needed.

### Stretch goals (optional)
- Add a “No gesture” class for robustness.
- Relative coordinate features (wrist-centered, palm-scaled) for +1–2% accuracy.
- Multi-hand support; dynamic gestures; export lightweight model for mobile.

