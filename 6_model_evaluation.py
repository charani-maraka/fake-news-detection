
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load cleaned dataset
data = pd.read_csv("dataset/cleaned_news.csv")

# Separate input and output
X = data["cleaned_text"]
y = data["label"]

# Split dataset (same settings as model training)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load trained model
model = joblib.load("fake_news_model.pkl")

# Predict test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("MODEL EVALUATION RESULTS")
print("------------------------")

print("Accuracy:", accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
