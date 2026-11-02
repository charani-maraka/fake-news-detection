
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
data = pd.read_csv("dataset/cleaned_news.csv")

# Count REAL and FAKE news
news_count = data["label"].value_counts()

print("News Distribution:")
print(news_count)

# Create bar chart
plt.figure(figsize=(7, 5))

sns.countplot(data=data, x="label")

plt.title("Fake News and Real News Distribution")
plt.xlabel("News Category")
plt.ylabel("Number of Articles")

plt.show()
