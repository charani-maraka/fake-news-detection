
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load cleaned dataset
data = pd.read_csv("dataset/cleaned_news.csv")

# Select input and output
X = data["cleaned_text"]
y = data["label"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create machine learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train the model
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "fake_news_model.pkl")

print("Model Training Completed Successfully!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Model saved successfully!")
