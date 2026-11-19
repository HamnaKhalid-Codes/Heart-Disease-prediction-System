
from flask import Flask, render_template, request
import pandas as pd
import joblib

# Initialize Flask application
app = Flask(__name__)

# Load the saved machine learning model
model = joblib.load("heart_disease_model.pkl")


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction route

@app.route("/predict", methods=["POST"])
def predict():

    features = [
        float(request.form["age"]),
        float(request.form["sex"]),
        float(request.form["cp"]),
        float(request.form["trestbps"]),
        float(request.form["chol"]),
        float(request.form["fbs"]),
        float(request.form["restecg"]),
        float(request.form["thalach"]),
        float(request.form["exang"]),
        float(request.form["oldpeak"]),
        float(request.form["slope"]),
        float(request.form["ca"]),
        float(request.form["thal"])
    ]

    columns = [
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalach", "exang",
        "oldpeak", "slope", "ca", "thal"
    ]

    patient_data = pd.DataFrame([features], columns=columns)

    # Generate prediction
    prediction = model.predict(patient_data)[0]

    # Generate probabilities
    probabilities = model.predict_proba(patient_data)[0]

    # Display result
    if prediction == 1:
        result = "Heart disease detected"
        result_class = "positive"
    else:
        result = "No heart disease detected"
        result_class = "negative"

    # Display prediction on webpage
    return render_template(
        "index.html",
        prediction=result,
        result_class=result_class,
        no_disease=round(probabilities[0] * 100, 2),
        heart_disease=round(probabilities[1] * 100, 2)
    )
if __name__ == "__main__":
    app.run(debug=True)