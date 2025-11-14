from flask import Flask, jsonify
from flask_cors import CORS
import sys
import os

# Add apps directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'apps'))

app = Flask(__name__)
CORS(app)

# Import and register blueprints
from apps.nlp_sentiment.nlp_blueprint import nlp_bp

app.register_blueprint(nlp_bp)

# Add more blueprints here as you create more apps:
# from apps.app2.app2_blueprint import app2_bp
# app.register_blueprint(app2_bp)

@app.route('/')
def index():
    return jsonify({
        "message": "Portfolio API Hub",
        "apps": [
            {"name": "NLP Sentiment Analyzer", "endpoint": "/api/nlp"},
            # Add more apps here
        ]
    })

@app.route('/api/health')
def health():
    return jsonify({"status": "ok", "message": "Portfolio API is running"})

if __name__ == "__main__":
    app.run(debug=True)
