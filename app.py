from flask import Flask, request, jsonify
import joblib
import nltk
import numpy as np
from flask_cors import CORS


# Download NLTK data if not already present
nltk.download("punkt")
nltk.download("stopwords")

app = Flask(__name__)
CORS(app)

# Load the trained pipeline (includes TF-IDF + model)
pipeline = joblib.load("sentiment_pipeline.pkl")

# Optional: Map score to gauge labels
def get_gauge_label(prob):
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

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    review = request.args.get("review") if request.method == "GET" else request.json.get("review")

    if not review:
        return jsonify({"error": "No review provided"}), 400

    # Make prediction
    prediction = pipeline.predict([review])[0]

    # Optional: Sentiment score for gauge (uncomment below if needed)
    score = pipeline.decision_function([review])[0]
    prob = 1 / (1 + np.exp(-score))  # Convert to probability
    gauge_label = get_gauge_label(prob)

    sentiment = "Positive" if prediction == 1 else "Negative"

    return jsonify({
        "sentiment": sentiment,
        "score": float(prob),
        "gauge": gauge_label
    })

if __name__ == "__main__":
    app.run(debug=True)
