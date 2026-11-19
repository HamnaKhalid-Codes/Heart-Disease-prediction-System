
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# STEP 1: LOAD DATASET
# ==========================================

heart = fetch_openml(
    data_id=43398,
    target_column="target",
    as_frame=True
)

X = heart.data
y = heart.target.astype(int)

df = pd.DataFrame(X)
df["target"] = y

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget percentage distribution:")
print(y.value_counts(normalize=True) * 100)


# ==========================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================

# Target distribution
sns.countplot(x=y)
plt.title("Heart Disease Class Distribution")
plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
plt.ylabel("Number of Patients")
plt.show()


# Statistical summary
print("\nStatistical Summary:")
print(df.describe())


# Age distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="age", bins=20, kde=True)
plt.title("Distribution of Patient Age")
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.show()


# Cholesterol distribution
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["chol"])
plt.title("Cholesterol Distribution")
plt.xlabel("Cholesterol")
plt.show()


# Cholesterol vs heart disease
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="target", y="chol")
plt.title("Cholesterol Levels vs Heart Disease")
plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
plt.ylabel("Cholesterol")
plt.show()


# Cholesterol outlier detection
for group in [0, 1]:

    group_data = df[df["target"] == group]["chol"]

    Q1 = group_data.quantile(0.25)
    Q3 = group_data.quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = group_data[
        (group_data < lower_limit) |
        (group_data > upper_limit)
    ]

    print(f"\nGroup {group} - Potential outliers:")
    print(outliers.values)


# Correlation heatmap
corr = df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Heart Disease Correlation Heatmap")
plt.show()


# ==========================================
# STEP 3: DATA PREPROCESSING
# ==========================================

X = df.drop("target", axis=1)
y = df["target"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)


# ==========================================
# STEP 4: TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining features:", X_train.shape)
print("Testing features:", X_test.shape)


# ==========================================
# STEP 5: FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# STEP 6: LOGISTIC REGRESSION
# ==========================================

lr_model = LogisticRegression(random_state=42)

lr_model.fit(X_train_scaled, y_train)

lr_pred = lr_model.predict(X_test_scaled)

lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)

print("\nLOGISTIC REGRESSION RESULTS")
print("Accuracy:", lr_accuracy * 100, "%")
print("Precision:", lr_precision)
print("Recall:", lr_recall)
print("F1-Score:", lr_f1)


# Confusion matrix for Logistic Regression
cm = confusion_matrix(y_test, lr_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Disease", "Heart Disease"]
)

display.plot(cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.show()


# ==========================================
# STEP 7: RANDOM FOREST CLASSIFIER
# ==========================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print("\nRANDOM FOREST RESULTS")
print("Accuracy:", rf_accuracy * 100, "%")
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("F1-Score:", rf_f1)


# ==========================================
# STEP 8: DECISION TREE CLASSIFIER
# ==========================================

dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred)
dt_recall = recall_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred)

print("\nDECISION TREE RESULTS")
print("Accuracy:", dt_accuracy * 100, "%")
print("Precision:", dt_precision)
print("Recall:", dt_recall)
print("F1-Score:", dt_f1)


# ==========================================
# STEP 9: K-NEAREST NEIGHBORS
# ==========================================

knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train_scaled, y_train)

knn_pred = knn_model.predict(X_test_scaled)

knn_accuracy = accuracy_score(y_test, knn_pred)
knn_precision = precision_score(y_test, knn_pred)
knn_recall = recall_score(y_test, knn_pred)
knn_f1 = f1_score(y_test, knn_pred)

print("\nKNN RESULTS")
print("Accuracy:", knn_accuracy * 100, "%")
print("Precision:", knn_precision)
print("Recall:", knn_recall)
print("F1-Score:", knn_f1)


# ==========================================
# STEP 10: FINAL MODEL COMPARISON
# ==========================================

comparison = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Random Forest",
        "Decision Tree",
        "KNN"
    ],

    "Accuracy": [
        lr_accuracy,
        rf_accuracy,
        dt_accuracy,
        knn_accuracy
    ],

    "Precision": [
        lr_precision,
        rf_precision,
        dt_precision,
        knn_precision
    ],

    "Recall": [
        lr_recall,
        rf_recall,
        dt_recall,
        knn_recall
    ],

    "F1-Score": [
        lr_f1,
        rf_f1,
        dt_f1,
        knn_f1
    ]
})

comparison = comparison.sort_values(
    by="F1-Score",
    ascending=False
)

print("\nFINAL MODEL COMPARISON")
print(comparison.to_string(index=False))

joblib.dump(rf_model, "heart_disease_model.pkl")

print("\nRandom Forest model saved successfully!")



# Load saved model
loaded_model = joblib.load("heart_disease_model.pkl")

# Make predictions using saved model
saved_model_pred = loaded_model.predict(X_test)

print("\nPredictions using saved model:")
print(saved_model_pred[:10])