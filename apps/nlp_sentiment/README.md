# IMDB Sentiment Analyzer

Natural Language Processing application for sentiment analysis on movie reviews.

## Overview

- **Endpoint**: `/api/nlp/predict`
- **Method**: POST
- **Input**: `{"review": "Your movie review text"}`
- **Output**:
  ```json
  {
    "sentiment": "Positive" or "Negative",
    "score": 0.8234,
    "gauge": "Good"
  }
  ```

## Model Details

- **Dataset**: IMDB 50K movie reviews
- **Preprocessing**: HTML removal, lowercasing, stopword filtering, punctuation removal
- **Vectorization**: TF-IDF (unigrams + bigrams)
- **Classifier**: Linear SVC
- **Accuracy**: ~90%

## Files

- `nlp_blueprint.py`: Flask Blueprint with prediction endpoint
- `models/sentiment_pipeline.pkl`: Trained ML pipeline
- `models/tfidf_vectorizer.pkl`: Text vectorizer
- `models/sentiment_model.pkl`: Classifier model
- `notebook.ipynb`: Training notebook with full analysis
- `data/`: Dataset files (requires Kaggle API key)
- `plots/`: Confusion matrix and distribution visualizations

## Local Testing

```python
# From project root
python app.py

# Test endpoint
curl -X POST http://localhost:5000/api/nlp/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "This movie was amazing!"}'
```
