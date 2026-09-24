# Speech Emotion Recognition using Machine Learning

## Overview

This repository contains an end-to-end audio processing and machine learning pipeline developed for Task 2: Emotion Recognition from Speech, part of the CodeAlpha Machine Learning Internship.

The objective is to classify human emotional states (happy, sad, angry, neutral, and others) directly from speech audio files using signal processing and ensemble classification techniques.

---

## Dataset Details

The model uses the RAVDESS dataset (Ryerson Audio-Visual Database of Emotional Speech and Song):

- **Audio Format:** `.wav` speech samples recorded by professional actors
- **Target Categories:** Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised
- **Filename Mapping:** Emotion labels are extracted directly from the standardized RAVDESS identifier structure (e.g. `03-01-05-01-01-01-01.wav`, where the third segment identifies the emotion category)

---

## Feature Extraction & Architecture

### Audio Feature Engineering (MFCCs)

Raw time-domain audio signals are processed with Librosa to extract Mel-Frequency Cepstral Coefficients (MFCCs):

- **Extraction:** 40 MFCC channels are computed across overlapping temporal frames
- **Dimensionality Reduction:** Summarized by averaging along the time axis to produce a fixed 40-dimensional feature vector per clip, reducing acoustic variance while preserving core timbral and pitch characteristics

### Pipeline Structure

- **`extract_mfcc_features()`** — Loads raw audio, applies sampling offsets, and computes mean MFCC vectors
- **`load_ravdess_dataset()`** — Recursively parses audio file paths, maps emotion identifiers, and builds the feature matrix `X` and target vector `y`
- **`evaluate_emotion_model()`** — Outputs accuracy, per-class precision and recall, and a confusion matrix visualization

---

## Model & Bias-Variance Trade-Off

- **Baseline Classifier:** Random Forest Classifier (100 estimators)
- **Trade-off Rationale:** Taking temporal means of MFCC features creates a tabular representation well-suited to Random Forest. Bagging averages predictions across uncorrelated decision trees, reducing high variance on smaller acoustic sample sizes without requiring a deep learning architecture.
- **Future Work:** Moving to sequence models (1D-CNN or LSTM) would capture pitch contours and temporal transitions, which should improve performance on larger speech corpora.

---

## Project Structure

```text
CodeAlpha_Emotion_Recognition/
├── data/                       # RAVDESS .wav files (organized in actor subfolders)
├── emotion_recognition.ipynb   # Main Jupyter Notebook
└── README.md                   # Project documentation
```

---

## Installation & Setup

### Prerequisites

Ensure Python 3.x and the audio processing dependencies are installed:

```bash
pip install librosa soundfile numpy scikit-learn seaborn matplotlib
```

### Execution

Place the RAVDESS audio files inside the `data/` directory, then open `emotion_recognition.ipynb` in VS Code or Jupyter and run the cells top to bottom.

The notebook extracts MFCC features from every `.wav` file, trains the classifier, and prints the evaluation report alongside a confusion matrix plot.

---

## Author & Acknowledgments

- **Developer:** Wireko Fosu Eric
- **Program:** Machine Learning Internship at CodeAlpha
