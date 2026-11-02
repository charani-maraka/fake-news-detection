import pandas as pd
import re

# Load dataset
data = pd.read_csv("dataset/fake_news.csv")

# Function to clean text
def clean_text(text):

    # Convert text into lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# Apply preprocessing
data["cleaned_text"] = data["text"].apply(clean_text)

# Display original and cleaned text
print(data[["text", "cleaned_text"]].head())

# Save cleaned dataset
data.to_csv("dataset/cleaned_news.csv", index=False)

print("\nData preprocessing completed!")