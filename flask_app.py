from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# -----------------------------
# Load Model
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "diet_recommendation_system (1).pkl"
)

with open(MODEL_PATH, "rb") as file:
    saved_objects = pickle.load(file)

model = saved_objects["model"]
disease_encoder = saved_objects["disease_encoder"]
activity_encoder = saved_objects["activity_encoder"]
diet_encoder = saved_objects["diet_encoder"]

# -----------------------------
# Home Page
# -----------------------------
@app.route("/")
def home():
    return render_template(
        "index.html",
        diseases=list(disease_encoder.classes_),
        activities=list(activity_encoder.classes_)
    )

# -----------------------------
# Prediction
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    try:

        age = int(request.form["age"])
        bmi = float(request.form["bmi"])

        disease = request.form["disease"]
        activity = request.form["activity"]

        disease_encoded = disease_encoder.transform([disease])[0]
        activity_encoded = activity_encoder.transform([activity])[0]

        sample = np.array([
            [age, bmi, disease_encoded, activity_encoded]
        ])

        prediction = model.predict(sample)

        diet = diet_encoder.inverse_transform(prediction)[0]

        return render_template(
            "index.html",
            prediction=diet,
            diseases=list(disease_encoder.classes_),
            activities=list(activity_encoder.classes_)
        )

    except Exception as e:
        return f"Error: {str(e)}"

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
