
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("heart_disease_model.pkl")

print("===================================")
print("   HEART DISEASE PREDICTION SYSTEM")
print("===================================")

# Collect patient information
age = float(input("Enter age: "))
sex = float(input("Enter sex (1 = Male, 0 = Female): "))
cp = float(input("Enter chest pain type (0-3): "))
trestbps = float(input("Enter resting blood pressure: "))
chol = float(input("Enter cholesterol level: "))
fbs = float(input("Enter fasting blood sugar (1 = True, 0 = False): "))
restecg = float(input("Enter resting ECG result (0-2): "))
thalach = float(input("Enter maximum heart rate: "))
exang = float(input("Enter exercise-induced angina (1 = Yes, 0 = No): "))
oldpeak = float(input("Enter oldpeak: "))
slope = float(input("Enter slope (0-2): "))
ca = float(input("Enter number of major vessels (0-4): "))
thal = float(input("Enter thal value (0-3): "))

# Arrange inputs in the same feature order used during training
patient_data = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal
}])

# Make prediction
prediction = model.predict(patient_data)

probabilities = model.predict_proba(patient_data)[0]

print("\nPrediction Probabilities:")
print(f"No Heart Disease: {probabilities[0] * 100:.2f}%")
print(f"Heart Disease: {probabilities[1] * 100:.2f}%")
# Display result
print("\n===================================")

if prediction[0] == 1:
    print("Prediction: Heart disease detected")
else:
    print("Prediction: No heart disease detected")

print("===================================")
