
import joblib
import re

# Load trained model
model = joblib.load("fake_news_model.pkl")

# Function to clean news text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Get news article from user
news = input("Enter a news article: ")

# Clean the news
cleaned_news = clean_text(news)

# Predict news category
prediction = model.predict([cleaned_news])

# Display result
print("\nPrediction Result:")

if prediction[0] == "FAKE":
    print("This news is predicted as FAKE.")
else:
    print("This news is predicted as REAL.")
