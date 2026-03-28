import pickle
import re
from flask import Flask, request, render_template_string

# Load the pre-trained model (pipeline)
with open('imdb_sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

# HTML template embedded as a string
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>IMDB Sentiment Analysis</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; text-align: center; }
        textarea { width: 80%; height: 150px; margin-bottom: 20px; }
        .result { margin-top: 20px; font-weight: bold; }
        .error { color: red; }
        .positive { color: green; }
        .negative { color: red; }
    </style>
</head>
<body>
    <h1>IMDB Movie Review Sentiment Analysis</h1>
    <form action="/predict" method="post">
        <textarea name="review" placeholder="Enter your movie review here...">{{ review if review else '' }}</textarea><br>
        <input type="submit" value="Analyze Sentiment">
    </form>

    {% if sentiment %}
        <div class="result">
            <p><strong>Your review:</strong> {{ review }}</p>
            <p><strong>Sentiment:</strong>
                <span class="{{ 'positive' if sentiment == 'Positive' else 'negative' }}">{{ sentiment }}</span>
            </p>
            <p><strong>Confidence:</strong> {{ confidence }}%</p>
        </div>
    {% endif %}

    {% if error %}
        <p class="error">{{ error }}</p>
    {% endif %}
</body>
</html>
'''

def preprocess_text(text):
    """Basic text cleaning – matches the vectorizer's expectations."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Keep only letters and spaces (optional, the vectorizer's token pattern handles this)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form.get('review', '')
    if not review.strip():
        return render_template_string(HTML_TEMPLATE, error='Please enter a review.')

    # Preprocess the input
    processed_review = preprocess_text(review)

    # Predict sentiment and probability
    prediction = model.predict([processed_review])[0]
    proba = model.predict_proba([processed_review])[0]

    sentiment = 'Positive' if prediction == 1 else 'Negative'
    confidence = round(proba[prediction] * 100, 2)

    return render_template_string(HTML_TEMPLATE,
                                   review=review,
                                   sentiment=sentiment,
                                   confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)