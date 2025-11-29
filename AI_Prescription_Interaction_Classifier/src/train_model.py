
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "drug_interactions.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

def load_data():
    # Load CSV with at least: drug1, drug2, description, severity
    df = pd.read_csv(DATA_PATH)

    # Example mapping if your dataset uses Minor/Moderate/Major.
    # Uncomment and adjust to match your CSV if needed.
    # severity_map = {
    #     "Minor": "Safe",
    #     "Moderate": "Moderate",
    #     "Major": "Severe"
    # }
    # df["severity"] = df["severity"].map(severity_map)
    # df = df.dropna(subset=["severity"])

    # Build combined text field
    df["text"] = (
        df["drug1"].astype(str) + " " +
        df["drug2"].astype(str) + " " +
        df["description"].astype(str)
    )

    X = df["text"]
    y = df["severity"]  # values: Safe, Moderate, Severe
    return X, y

def save_confusion_matrix(cm, labels, path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def main():
    print(f"Loading data from: {DATA_PATH}")
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_features=10000
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        n_jobs=-1
    )

    print("Training model...")
    model.fit(X_train_vec, y_train)

    print("\nEvaluating on test set...")
    y_pred = model.predict(X_test_vec)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=["Safe", "Moderate", "Severe"])
    print("Confusion matrix:")
    print(cm)

    cm_path = BASE_DIR / "confusion_matrix.png"
    save_confusion_matrix(cm, ["Safe", "Moderate", "Severe"], cm_path)
    print(f"\nSaved confusion matrix image to {cm_path}")

    model_path = MODELS_DIR / "severity_classifier.pkl"
    vec_path = MODELS_DIR / "tfidf_vectorizer.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Saved model to {model_path}")
    print(f"Saved vectorizer to {vec_path}")

if __name__ == "__main__":
    main()
