import pandas as pd

# Load dataset
data = pd.read_csv("dataset/fake_news.csv")

# Display first 5 records
print(data.head())

# Display dataset information
print("\nDataset Information:")
print(data.info())

# Display dataset shape
print("\nDataset Shape:")
print(data.shape)

# Count REAL and FAKE news
print("\nNews Distribution:")
print(data["label"].value_counts())
