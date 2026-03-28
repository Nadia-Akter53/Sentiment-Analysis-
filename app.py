import streamlit as st
import pickle
import re

# Load the model
@st.cache_resource
def load_model():
    with open('imdb_sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

def preprocess_text(text):
    """Basic text cleaning – matches the vectorizer's expectations."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Keep only letters and spaces (optional, the vectorizer's token pattern handles this)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()

# Streamlit UI
st.set_page_config(page_title="IMDB Sentiment Analysis", page_icon="🎬")
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review below and find out if it's **positive** or **negative**.")

review = st.text_area("Your review:", height=200)

if st.button("Analyze"):
    if not review.strip():
        st.warning("Please enter a review.")
    else:
        processed_review = preprocess_text(review)
        prediction = model.predict([processed_review])[0]
        proba = model.predict_proba([processed_review])[0]

        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = round(proba[prediction] * 100, 2)

        st.markdown("---")
        st.subheader("Results")
        st.write(f"**Review:** {review}")
        if sentiment == "Positive":
            st.success(f"**Sentiment:** {sentiment} (confidence: {confidence}%)")
        else:
            st.error(f"**Sentiment:** {sentiment} (confidence: {confidence}%)")
