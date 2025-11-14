# Application Portfolio Hub
## Preston Brownlee

**Live Application**: [https://imdb-sentiment-frontend-ten.vercel.app/](https://imdb-sentiment-frontend-ten.vercel.app/)

This is a portfolio hub containing multiple machine learning and data science applications. The landing page provides access to all individual projects through a unified interface.

## System Architecture

- **Single Flask Backend** with lazy-loaded Blueprints - Only loads resources when specific apps are accessed (saves memory on free tier)
- **React Frontend** with React Router - Single-page app with multiple routes for different projects
- **Modular Structure** - Each app lives in its own folder with independent models and logic

**Deployment:**
- Backend: Render.com (Free tier) – [https://imdb-sentiment-backend.onrender.com/](https://imdb-sentiment-backend.onrender.com/)
- Frontend: Vercel (Free tier) – [https://imdb-sentiment-frontend-ten.vercel.app/](https://imdb-sentiment-frontend-ten.vercel.app/)

---

## Current Applications

### 1. IMDB Sentiment Analyzer
- **Goal**: Classify IMDB movie reviews as positive or negative
- **Dataset**: IMDB movie review dataset (balanced ~50/50 positive/negative)
- **Model Pipeline**: Text preprocessing (punctuation removal, stop‑word filtering), TF‑IDF vectorization (unigrams + bigrams), Linear SVC classifier
- **Performance**: Achieved ~90% accuracy on test set
- **Endpoint**: `/api/nlp/predict`
- **Route**: `/nlp-sentiment`

More apps coming soon!

---

## Setup Instructions

> Note: This application is fully deployed and accessible online. It may take a minute or so to run the first time online since I'm using the free version of render. Once it runs the first time it should run quickly after that. 
> You can use the live version here:  
> [https://imdb-sentiment-frontend-ten.vercel.app/](https://imdb-sentiment-frontend-ten.vercel.app/)

> If you'd prefer to run the project locally (or if you're offline), follow the steps below to launch the backend and frontend manually from your machine.

---

### 1. Download & Unzip

- Extract the folder to your preferred location

---

### 2. Backend Setup (Flask API)

1. Create and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate     # On Windows: venv\Scripts\activate
    ```

2. Install Python dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the Flask server:

    ```bash
    python app.py
    ```

- The backend will be live at: [http://localhost:5000](http://localhost:5000)

---

### 3. Frontend Setup (React App)

1. Open a new terminal and navigate into the frontend directory:

    ```bash
    cd frontend
    ```

2. Install Node.js dependencies:

    ```bash
    npm install
    ```

3. Start the React development server:

    ```bash
    npm start
    ```

- The frontend will be live at: [http://localhost:3000](http://localhost:3000)
- You'll see the portfolio landing page with cards for each app
- Click on any app to navigate to it

---

Once both servers are running, you can open the app in your browser and interact with the sentiment analyzer in real time.
---

---

## Project Structure

```
NLPApplication/
├── app.py                    # Main Flask app with blueprint registration
├── apps/                     # Individual applications
│   ├── nlp-sentiment/       # IMDB Sentiment Analyzer
│   │   ├── nlp_blueprint.py # Flask Blueprint (lazy-loaded)
│   │   ├── models/          # ML models (.pkl files)
│   │   ├── notebook.ipynb   # Training notebook
│   │   ├── data/            # Dataset files
│   │   └── plots/           # Visualizations
│   └── [future apps]/       # Add more apps here
├── frontend/                 # React frontend
│   └── src/
│       ├── App.js           # Main router
│       ├── pages/
│       │   ├── Home.js      # Portfolio landing page
│       │   └── NLPSentiment/  # NLP app component
│       └── ...
└── requirements.txt

```

---

## Adding New Applications

See `apps/README.md` for detailed instructions on adding new applications to the portfolio.

---

## NLP Sentiment Analyzer - Training Details

Model development and training are in `apps/nlp-sentiment/notebook.ipynb`:

This notebook handles:

-  **Kaggle Dataset Access**:  
  The IMDB movie review dataset is pulled directly from Kaggle using the Kaggle API.  
  Note: I’ve removed my personal `kaggle.json` key for privacy.  
  To download the data:
  1. Go to Kaggle - https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
  2. Download your own `kaggle.json` API key
  3. Place it inside a folder named `data/` in the project root
  4. The notebook will detect the key and download the dataset

-  **Data Cleaning**:  
  - HTML tag stripping  
  - Lowercasing  
  - Stopword removal  
  - Punctuation removal  
  - Tokenization

-  **Vectorization**:  
  - TF-IDF (unigrams + bigrams)  
  - 80/10/10 train/validation/test split

-  **Model Training**:  
  - Trains both a Logistic Regression model and a Linear SVC model  
  - Evaluates performance (accuracy, precision, recall, F1 score)  
  - Generates confusion matrix and class distribution plots

-  **Model Saving**:  
  - The final pipeline (TF-IDF + model) is serialized and saved to disk:  
    - `sentiment_pipeline.pkl`  
    - `tfidf_vectorizer.pkl`

Once complete, these files are used by the Flask backend to serve real-time predictions.

---

## Known Issues & Notes

- The backend (hosted on Render) may take 30–60 seconds to respond on the first request if inactive.
- You must add your own `kaggle.json` API key to use the notebook or download the dataset locally.
- Make sure both frontend and backend URLs match if deploying separately.

