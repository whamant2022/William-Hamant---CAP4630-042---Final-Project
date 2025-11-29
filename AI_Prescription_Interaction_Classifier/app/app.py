
from flask import Flask, render_template, request
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

model_path = MODELS_DIR / "severity_classifier.pkl"
vec_path = MODELS_DIR / "tfidf_vectorizer.pkl"

# Load trained model + vectorizer
with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(vec_path, "rb") as f:
    vectorizer = pickle.load(f)

app = Flask(__name__)

LABEL_TO_MESSAGE = {
    "Safe": "Safe / Minor – no clinically significant interaction expected.",
    "Moderate": "Moderate – use with caution and monitor the patient.",
    "Severe": "Severe / Major – combination generally should be avoided."
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        drug1 = request.form.get("drug1", "").strip()
        drug2 = request.form.get("drug2", "").strip()

        if not drug1 or not drug2:
            error = "Please enter both drug names."
            return render_template("index.html", error=error)

        text = f"{drug1} {drug2}"
        X = vectorizer.transform([text])
        pred = model.predict(X)[0]
        message = LABEL_TO_MESSAGE.get(pred, "")

        return render_template(
            "result.html",
            drug1=drug1,
            drug2=drug2,
            severity=pred,
            message=message
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
