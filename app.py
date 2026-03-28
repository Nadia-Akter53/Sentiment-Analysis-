import streamlit as st
import pickle
import re
import os

# ----------------------------------------------------------------------
# Load the model (cached to avoid reloading on every interaction)
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = 'imdb_sentiment_model.pkl'
    if not os.path.exists(model_path):
        st.error(
            f"Model file '{model_path}' not found. "
            "Please make sure it is in the same folder as this script."
        )
        st.stop()
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# ----------------------------------------------------------------------
# Text preprocessing – matches the vectorizer's expectations
# ----------------------------------------------------------------------
def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Keep only letters and spaces (optional – the vectorizer's token pattern handles it)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()

# ----------------------------------------------------------------------
# Custom CSS to style the app
# ----------------------------------------------------------------------
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 1rem;
    }
    
    /* Title */
    .title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    
    /* Review text area */
    .stTextArea textarea {
        font-size: 1rem;
        border-radius: 8px;
        border: 1px solid #bdc3c7;
        padding: 12px;
    }
    
    /* Button */
    .stButton button {
        background-color: #3498db;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        width: 100%;
        transition: 0.3s;
    }
    .stButton button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
    }
    
    /* Result box */
    .result-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 2rem;
        border-left: 5px solid;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .positive {
        border-left-color: #2ecc71;
    }
    .negative {
        border-left-color: #e74c3c;
    }
    .sentiment-positive {
        color: #2ecc71;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #e74c3c;
        font-weight: bold;
    }
    .confidence {
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# HTML/JavaScript to add a simple animation or effect (optional)
# ----------------------------------------------------------------------
st.components.v1.html("""
<script>
    // This script runs once when the page loads (optional)
    console.log("App loaded!");
</script>
""", height=0)

# ----------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------
st.markdown('<div class="title">🎬 IMDB Movie Review Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter a movie review and see if it\'s positive or negative</div>', unsafe_allow_html=True)

review = st.text_area("", placeholder="Type or paste your review here...", height=200)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button("Analyze Sentiment", use_container_width=True)

if analyze_button:
    if not review.strip():
        st.warning("Please enter a review.")
    else:
        # Preprocess and predict
        processed_review = preprocess_text(review)
        prediction = model.predict([processed_review])[0]
        proba = model.predict_proba([processed_review])[0]

        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = round(proba[prediction] * 100, 2)

        # Determine box style
        box_class = "positive" if sentiment == "Positive" else "negative"
        sentiment_class = "sentiment-positive" if sentiment == "Positive" else "sentiment-negative"

        # Display result using custom HTML
        st.markdown(f"""
        <div class="result-box {box_class}">
            <div style="font-weight: bold; margin-bottom: 0.5rem;">Your Review:</div>
            <div style="background-color: white; padding: 0.75rem; border-radius: 5px; margin-bottom: 1rem;">
                {review}
            </div>
            <div style="font-size: 1.2rem;">
                <strong>Sentiment:</strong>
                <span class="{sentiment_class}">{sentiment}</span>
            </div>
            <div class="confidence">
                <strong>Confidence:</strong> {confidence}%
            </div>
        </div>
        """, unsafe_allow_html=True)
