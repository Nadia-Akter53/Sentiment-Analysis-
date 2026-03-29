import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import pickle
import re
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

# ----------------------------------------------------------------------
# Custom CSS
# ----------------------------------------------------------------------
st.markdown("""
<style>
    .main { padding: 0rem 1rem; }
    
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
        display: inline-block;
    }
    
    .prediction-box {
        text-align: center;
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        animation: fadeIn 0.5s ease-in;
    }
    
    .sentiment-positive {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.3);
    }
    
    .sentiment-negative {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        box-shadow: 0 10px 30px rgba(239, 68, 68, 0.3);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .probability-bar {
        background: rgba(255,255,255,0.3);
        border-radius: 10px;
        height: 10px;
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .probability-fill {
        height: 100%;
        background: white;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    .badge-positive {
        background: #10b981;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
    }
    
    .badge-negative {
        background: #ef4444;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Database functions
# ----------------------------------------------------------------------
def init_database():
    conn = sqlite3.connect('sentiment.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_text TEXT,
            sentiment INTEGER,
            sentiment_label TEXT,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(review_text, sentiment, confidence):
    conn = sqlite3.connect('sentiment.db')
    cursor = conn.cursor()
    sentiment_label = "Positive" if sentiment == 1 else "Negative"
    cursor.execute('''
        INSERT INTO predictions (review_text, sentiment, sentiment_label, confidence)
        VALUES (?, ?, ?, ?)
    ''', (review_text, sentiment, sentiment_label, confidence))
    conn.commit()
    conn.close()

def get_all_predictions():
    conn = sqlite3.connect('sentiment.db')
    df = pd.read_sql("SELECT * FROM predictions ORDER BY created_at DESC", conn)
    conn.close()
    return df

def get_stats():
    df = get_all_predictions()
    if len(df) == 0:
        return 0, 0, 0, 0.0
    total = len(df)
    positive = len(df[df['sentiment'] == 1])
    negative = len(df[df['sentiment'] == 0])
    avg_conf = df['confidence'].mean() * 100
    return total, positive, negative, avg_conf

# ----------------------------------------------------------------------
# Model loading (using pickle with the exact filename)
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    # Note: filename includes a space before .pkl
    model_path = 'imdb_sentiment_model .pkl'
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file '{model_path}' not found. Please upload it to the same directory.")
        st.stop()
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.error("Possible reasons:\n"
                 "- The file is corrupted.\n"
                 "- The file was saved with a different Python version.\n"
                 "- The file exceeds GitHub's size limit and was not fully uploaded.\n\n"
                 "Try using Git LFS or re‑upload the model file.")
        st.stop()

# ----------------------------------------------------------------------
# Text preprocessing
# ----------------------------------------------------------------------
def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()

# ----------------------------------------------------------------------
# Initialize
# ----------------------------------------------------------------------
init_database()
model = load_model()

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966322.png", width=80)
    st.markdown("## 🎬 IMDB Sentiment Analysis")
    st.markdown("---")
    
    st.markdown("### 📊 Statistics")
    total_pred, pos_pred, neg_pred, avg_conf = get_stats()
    st.metric("Total Predictions", total_pred)
    st.metric("Positive Reviews", pos_pred)
    st.metric("Negative Reviews", neg_pred)
    st.metric("Avg Confidence", f"{avg_conf:.1f}%")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info(
        "This app uses a machine learning model trained on the IMDB dataset "
        "to classify movie reviews as **Positive** or **Negative**.\n\n"
        "Model: Multinomial Naive Bayes with CountVectorizer."
    )

# ----------------------------------------------------------------------
# Main content
# ----------------------------------------------------------------------
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.markdown("---")

# Stats row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{total_pred}</div>
        <div class="metric-label">Total Predictions</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{pos_pred}</div>
        <div class="metric-label">Positive Reviews</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{neg_pred}</div>
        <div class="metric-label">Negative Reviews</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{avg_conf:.1f}%</div>
        <div class="metric-label">Avg Confidence</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict Sentiment", "📜 Prediction History", "📊 Visualizations", "ℹ️ About"])

# Tab 1: Prediction
with tab1:
    st.markdown('<span class="section-header">Enter a Movie Review</span>', unsafe_allow_html=True)
    
    review = st.text_area(
        "Your review",
        height=200,
        placeholder="Type or paste a movie review here... e.g., 'This movie was absolutely fantastic! The acting was superb.'"
    )
    
    if st.button("🔮 ANALYZE SENTIMENT", type="primary", use_container_width=True):
        if not review.strip():
            st.warning("Please enter a review.")
        else:
            processed = preprocess_text(review)
            prediction = model.predict([processed])[0]
            proba = model.predict_proba([processed])[0]
            confidence = proba[prediction]
            
            # Save to database
            save_prediction(review, prediction, confidence)
            
            sentiment = "Positive" if prediction == 1 else "Negative"
            sentiment_class = "sentiment-positive" if prediction == 1 else "sentiment-negative"
            confidence_percent = confidence * 100
            
            # Show result
            st.markdown("---")
            st.markdown("## 📊 Prediction Result")
            
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-box {sentiment_class}">
                    <h1 style="font-size: 3rem;">😊 POSITIVE SENTIMENT</h1>
                    <p style="font-size: 1.5rem;">The review is <strong>{sentiment}</strong> with {confidence_percent:.1f}% confidence</p>
                    <div class="probability-bar">
                        <div class="probability-fill" style="width: {confidence_percent:.0f}%;"></div>
                    </div>
                    <p style="margin-top: 1rem;">👍 The model believes this review expresses a positive opinion about the movie.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box {sentiment_class}">
                    <h1 style="font-size: 3rem;">😞 NEGATIVE SENTIMENT</h1>
                    <p style="font-size: 1.5rem;">The review is <strong>{sentiment}</strong> with {confidence_percent:.1f}% confidence</p>
                    <div class="probability-bar">
                        <div class="probability-fill" style="width: {confidence_percent:.0f}%;"></div>
                    </div>
                    <p style="margin-top: 1rem;">👎 The model believes this review expresses a negative opinion about the movie.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Show confidence gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = confidence_percent,
                title = {'text': "Confidence Score"},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#10b981" if prediction == 1 else "#ef4444"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 100], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': confidence_percent
                    }
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)

# Tab 2: Prediction History
with tab2:
    st.markdown('<span class="section-header">Prediction History</span>', unsafe_allow_html=True)
    
    history_df = get_all_predictions()
    if len(history_df) > 0:
        display_df = history_df[['created_at', 'review_text', 'sentiment_label', 'confidence']].copy()
        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x*100:.1f}%")
        display_df.rename(columns={
            'created_at': 'Date & Time',
            'review_text': 'Review',
            'sentiment_label': 'Sentiment',
            'confidence': 'Confidence'
        }, inplace=True)
        st.dataframe(display_df, use_container_width=True, height=400)
        
        if st.button("🗑️ Clear All History", type="secondary"):
            conn = sqlite3.connect('sentiment.db')
            conn.execute("DELETE FROM predictions")
            conn.commit()
            conn.close()
            st.success("History cleared!")
            st.rerun()
    else:
        st.info("No predictions yet. Enter a review to see history!")

# Tab 3: Visualizations
with tab3:
    st.markdown('<span class="section-header">Visual Analytics</span>', unsafe_allow_html=True)
    
    history_df = get_all_predictions()
    if len(history_df) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution pie chart
            sentiment_counts = history_df['sentiment_label'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']
            fig1 = px.pie(sentiment_counts, values='Count', names='Sentiment',
                          title='Sentiment Distribution',
                          color='Sentiment',
                          color_discrete_map={'Positive': '#10b981', 'Negative': '#ef4444'})
            fig1.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Confidence distribution
            fig2 = px.histogram(history_df, x='confidence', nbins=20,
                               title='Confidence Score Distribution',
                               color_discrete_sequence=['#667eea'])
            fig2.update_layout(xaxis_title='Confidence', yaxis_title='Count')
            st.plotly_chart(fig2, use_container_width=True)
        
        # Timeline of predictions
        history_df['date'] = pd.to_datetime(history_df['created_at']).dt.date
        timeline = history_df.groupby(['date', 'sentiment_label']).size().reset_index(name='count')
        fig3 = px.line(timeline, x='date', y='count', color='sentiment_label',
                      title='Predictions Over Time',
                      color_discrete_map={'Positive': '#10b981', 'Negative': '#ef4444'})
        st.plotly_chart(fig3, use_container_width=True)
        
        # Word cloud placeholder
        st.markdown("#### 📝 Most Common Words in Positive Reviews")
        st.info("Word cloud feature can be added here using the wordcloud library.")
    else:
        st.info("No data yet. Make some predictions to see visualizations!")

# Tab 4: About
with tab4:
    st.markdown('<span class="section-header">About This App</span>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎬 IMDB Sentiment Analysis
    
    This application uses a machine learning model to classify movie reviews as **Positive** or **Negative**.
    
    **Model Details:**
    - **Algorithm:** Multinomial Naive Bayes
    - **Vectorizer:** CountVectorizer with unigrams
    - **Training Data:** IMDB Dataset of 50,000 movie reviews
    - **Accuracy:** ~86% on test set
    
    **How to Use:**
    1. Enter a movie review in the text box.
    2. Click "ANALYZE SENTIMENT".
    3. The model will predict whether the review is positive or negative.
    4. View the confidence score and analysis.
    5. All predictions are saved and can be reviewed in the History tab.
    
    **Data Source:** [IMDB Dataset of 50k Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
    
    **Technology Stack:**
    - Streamlit for the web interface
    - Scikit-learn for the ML model
    - Plotly for interactive visualizations
    - SQLite for storing predictions
    
    ---
    *Note: This is a demonstration project.*
    """)

st.markdown("---")
st.caption("🎬 IMDB Sentiment Analysis | Powered by Machine Learning | Built with Streamlit")
