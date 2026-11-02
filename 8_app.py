import streamlit as st
import joblib
import re

st.set_page_config(page_title="Fake News Detection", page_icon="📰")

model = joblib.load("fake_news_model.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

st.title("📰 Fake News Detection System")
st.subheader("Detect whether a news article is Fake or Real")

st.info("Enter a news article below and click Predict.")

news = st.text_area("Enter News Article:", height=150)

if st.button("🔍 Predict News"):
    if news.strip():
        cleaned_news = clean_text(news)
        prediction = model.predict([cleaned_news])

        if prediction[0] == "FAKE":
            st.error("⚠️ This news is predicted as FAKE!")
        else:
            st.success("✅ This news is predicted as REAL!")
    else:
        st.warning("Please enter a news article.")

st.caption("Mini Project: Fake News Detection using Machine Learning")