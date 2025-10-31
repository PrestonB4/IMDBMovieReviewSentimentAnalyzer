# IMDB Movie Review Sentiment Analyzer
## Preston Brownlee

This project is a Natural Language Processing (NLP) application that performs **sentiment analysis** on movie reviews from the IMDB dataset. It predicts whether a review expresses **positive** or **negative** sentiment, using classical machine learning techniques and Python libraries.

The model is also deployed in a React-based frontend for real-time sentiment prediction.

---

## Project Overview

- **Goal**: Classify IMDB movie reviews as either *positive* or *negative*.
- **Dataset**: `IMDB Dataset for Sentiment Analysis` from Kaggle (CSV format).
- **Model**: Trained using Logistic Regression and later enhanced using LinearSVC.
- **Accuracy**: Achieved over 90% on the test set.
- **Deployment**: React frontend with a sentiment gauge and prediction results.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/sentiment-analyzer.git
cd sentiment-analyzer
```

### 2. Install Python Requirements

It's best to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download the IMDB Dataset (via Kaggle)

1. Go to https://www.kaggle.com/account
2. Scroll to the API section and click "Create New API Token".
3. Move the `kaggle.json` file into the `data/` folder in this repo.
4. Use the download snippet from the Jupyter notebook:

```python
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
api.dataset_download_files(
    "columbine/imdb-dataset-sentiment-analysis-in-csv-format",
    path="data/",
    unzip=True
)
```

---

### 4. Start the Backend (Flask API)

```bash
python app.py
```

---

### 5. Start the Frontend (React App)

```bash
cd sentiment-frontend
npm install
npm start
```

The app will launch at: `http://localhost:3000`

---

## Project Folder Structure

```
sentiment-analyzer/
│
├── app.py                       # Flask backend API
├── notebook.ipynb              # Jupyter Notebook (full training pipeline)
├── requirements.txt            # Python dependencies
├── sentiment_model.pkl         # Logistic regression model
├── sentiment_pipeline.pkl      # Final LinearSVC pipeline
│
├── data/                       # Contains Kaggle credentials and dataset
│   ├── kaggle.json
│   ├── Train.csv
│   ├── Test.csv
│   └── Valid.csv
│
├── plots/                      # Output visuals for evaluation
│   ├── sentiment_distribution.png
│   └── confusion_matrix.png
│
└── sentiment-frontend/         # React frontend
    ├── App.js
    ├── App.css
    └── ...
```

---

## Requirements

You can install all backend dependencies with:

```bash
pip install -r requirements.txt
```

**requirements.txt**:

```
numpy
pandas
matplotlib
seaborn
nltk
scikit-learn
joblib
beautifulsoup4
kaggle
```

---

## Notebook

The complete model pipeline is documented in `notebook.ipynb`, which includes:
- Data loading and preprocessing
- EDA and visualizations
- Model training (Logistic Regression and SVM)
- Accuracy & evaluation (confusion matrix, classification report)
- Final model export

---

## Deployment

This app is ready for deployment:
- Frontend can be deployed on **Vercel**
- Backend can be hosted via **Render** or **Replit**

Links will be added when deployed!

---
