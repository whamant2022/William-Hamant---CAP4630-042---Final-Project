
# AI Prescription Interaction Risk Classifier

## Overview
This project builds a machine learning model that predicts the **severity of drug–drug interactions**
(Safe, Moderate, Severe) using text descriptions of known interactions.

Final project for **CAP 4630 – Intro to Artificial Intelligence (Dr. Imteaj)**.

## Dataset
- Public drug–drug interaction dataset (e.g., Kaggle DDI dataset).
- Each row: `drug1`, `drug2`, `description`, `severity`.
- Severity mapped into three classes: `Safe`, `Moderate`, `Severe`.

Place your cleaned CSV as:
```
data/drug_interactions.csv
```

## How to Run

1. (Optional) Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure your dataset CSV exists at:

   ```text
   data/drug_interactions.csv
   ```

4. Train the model and generate metrics:

   ```bash
   python -m src.train_model
   ```

   This will:
   - Print a classification report in the terminal.
   - Save `models/severity_classifier.pkl` and `models/tfidf_vectorizer.pkl`.
   - Save `confusion_matrix.png` for the slides.

5. Run the Flask app:

   ```bash
   cd app
   python app.py
   ```

6. Open your browser at `http://127.0.0.1:5000` and test different drug pairs.

## Presentation & Demo

- Slide deck: `AI_Prescription_Interaction_Classifier_Presentation.pptx`
- Demo video: (add your YouTube / Google Drive link here)
