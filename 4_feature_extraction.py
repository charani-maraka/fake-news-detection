
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the cleaned dataset
data = pd.read_csv("dataset/cleaned_news.csv")

# Select cleaned news text
news = data["cleaned_text"]

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words="english")

# Convert text into numerical features
features = tfidf.fit_transform(news)

# Display the results
print("Feature Extraction Completed!")

print("Feature Matrix Shape:", features.shape)

print("\nFirst 10 Feature Names:")
print(tfidf.get_feature_names_out()[:10])
