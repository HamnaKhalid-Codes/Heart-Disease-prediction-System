# ❤️ Heart Disease Prediction System

An end-to-end Machine Learning project that predicts the presence or absence of heart disease using 13 patient health features. It includes data analysis, model training, evaluation, and a Flask web application for making predictions.

> **Disclaimer:** This project is for educational purposes only and is not intended for medical diagnosis.

## 📊 Dataset

- **Dataset:** Cleveland Heart Disease Dataset
- **Source:** OpenML
- **Dataset ID:** `43398`
- **Records:** 303
- **Features:** 13
- **Target:** 0 (No Heart Disease), 1 (Heart Disease)

## 🔍 Data Analysis & Preprocessing

Exploratory Data Analysis was performed using Pandas, Matplotlib, and Seaborn.

Key activities:
- Dataset inspection and statistical analysis
- Target distribution and feature visualization
- Outlier detection using the IQR method
- Correlation analysis and heatmap
- 80/20 stratified train-test split
- Feature scaling using StandardScaler for Logistic Regression and KNN

## 🤖 Machine Learning Models

Four classification algorithms were trained and evaluated:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest

### Model Performance

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Random Forest | **83.61%** | 78.05% | 96.97% | **86.49%** |
| Logistic Regression | 80.33% | 76.92% | 90.91% | 83.33% |
| KNN | 80.33% | 76.92% | 90.91% | 83.33% |
| Decision Tree | 70.49% | 70.27% | 78.79% | 74.29% |

**Selected Model:** Random Forest Classifier, based on its performance on the current test split.

The trained model was saved using Joblib as `heart_disease_model.pkl`.

## 🌐 Flask Web Application

A browser-based prediction application was developed using Flask, HTML, and CSS.

**How it works:**

1. User enters 13 patient features.
2. Flask receives the form data.
3. Pandas prepares the input DataFrame.
4. The saved Random Forest model generates a prediction.
5. The application displays the predicted class and estimated class probabilities.

## 🏗️ Project Structure

```text
ML/
│
├── heart-disease-prediction.py
├── predict_heart_disease.py
├── app.py
├── heart_disease_model.pkl
├── pyproject.toml
├── uv.lock
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
└── README.md
```

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | Data visualization |
| Scikit-learn | Machine Learning |
| Joblib | Model saving and loading |
| Flask | Backend web framework |
| HTML & CSS | Frontend |
| uv | Dependency management |

## 🚀 Installation & Execution

### 1. Clone the repository

```bash
git clone https://github.com/HamnaKhalid-Codes/Heart-Disease-prediction-System.git
cd Heart-Disease-prediction-system 
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Train the model

```bash
uv run python heart-disease-prediction.py
```

### 4. Run standalone prediction

```bash
uv run python predict_heart_disease.py
```

### 5. Start the Flask application

```bash
uv run python app.py
```

Open your browser:

```text
http://127.0.0.1:5000
```

## 📚 Key Learning Outcomes

- Exploratory Data Analysis and feature analysis
- Data preprocessing and feature scaling
- Training and comparing classification algorithms
- Model evaluation using classification metrics
- Saving and loading trained ML models
- Building a prediction application with Flask
- Connecting a Machine Learning model to an HTML/CSS interface

## 🔮 Future Improvements

- Improve input validation and error handling
- Add model performance visualizations
- Improve the user interface
- Deploy the application online

---

**Note:** Model results are based on a single train-test split and are not evidence of clinical accuracy. Model probabilities are estimates, not medical risk assessments.
