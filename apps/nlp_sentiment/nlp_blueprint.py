from flask import Blueprint, request, jsonify
import joblib
import nltk
import numpy as np
import os

# Create blueprint
nlp_bp = Blueprint('nlp', __name__, url_prefix='/api/nlp')

# Lazy loading - pipeline loaded only when first used
_pipeline = None

def get_pipeline():
    """Lazy load the ML pipeline to save memory"""
    global _pipeline
    if _pipeline is None:
        # Download NLTK data if not already present
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download("punkt", quiet=True)

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download("stopwords", quiet=True)

        # Load the trained pipeline
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'sentiment_pipeline.pkl')
        _pipeline = joblib.load(model_path)

    return _pipeline

def get_gauge_label(prob):
    """Map score to gauge labels"""
    if prob < 0.2:
        return "Very Poor"
    elif prob < 0.4:
        return "Poor"
    elif prob < 0.6:
        return "Okay"
    elif prob < 0.8:
        return "Good"
    else:
        return "Very Good"

@nlp_bp.route('/predict', methods=['GET', 'POST'])
def predict():
    """Predict sentiment of a movie review"""
    review = request.args.get("review") if request.method == "GET" else request.json.get("review")

    if not review:
        return jsonify({"error": "No review provided"}), 400

    # Load pipeline only when needed
    pipeline = get_pipeline()

    # Make prediction
    prediction = pipeline.predict([review])[0]
    score = pipeline.decision_function([review])[0]
    prob = 1 / (1 + np.exp(-score))  # Convert to probability
    gauge_label = get_gauge_label(prob)
    sentiment = "Positive" if prediction == 1 else "Negative"

    return jsonify({
        "sentiment": sentiment,
        "score": float(prob),
        "gauge": gauge_label
    })

@nlp_bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "app": "NLP Sentiment Analyzer"})
