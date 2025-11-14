# Portfolio Hub - Changes Summary

## What Was Changed

Your single NLP app has been transformed into a **portfolio hub** that can host multiple applications!

### New Structure

```
Before:                          After:
├── app.py                      ├── app.py (main router)
├── sentiment_pipeline.pkl      ├── apps/
├── tfidf_vectorizer.pkl        │   ├── nlp-sentiment/
├── sentiment_model.pkl         │   │   ├── nlp_blueprint.py
├── notebook.ipynb              │   │   ├── models/
├── data/                       │   │   │   ├── sentiment_pipeline.pkl
├── plots/                      │   │   │   ├── tfidf_vectorizer.pkl
└── sentiment-frontend/         │   │   │   └── sentiment_model.pkl
    └── src/                    │   │   ├── notebook.ipynb
        └── App.js              │   │   ├── data/
                                │   │   └── plots/
                                └── sentiment-frontend/
                                    └── src/
                                        ├── App.js (router)
                                        └── pages/
                                            ├── Home.js (landing page)
                                            └── NLPSentiment/
                                                └── NLPSentiment.js
```

## Key Features

### 1. Landing Page
- Beautiful portfolio home screen with app cards
- Click any card to navigate to that app
- Easy to add new apps to the grid

### 2. Lazy Loading Backend
- **Memory Efficient**: Models only load when their endpoints are called
- **Single Backend**: All apps share one Flask instance
- **Blueprint Architecture**: Each app is isolated in its own Blueprint

### 3. React Router
- Smooth navigation between apps
- Back button on each app to return home
- No page reloads - SPA experience

## What You Need to Do

### Option 1: Test Locally First

1. **Update the API endpoint for local testing**:

   Edit `sentiment-frontend/src/pages/NLPSentiment/NLPSentiment.js` line 14:
   ```javascript
   // Change from:
   const res = await fetch('https://imdb-sentiment-backend.onrender.com/api/nlp/predict', {

   // To:
   const res = await fetch('http://localhost:5000/api/nlp/predict', {
   ```

2. **Start the backend**:
   ```bash
   python app.py
   ```

3. **Start the frontend** (in separate terminal):
   ```bash
   cd sentiment-frontend
   npm start
   ```

4. **Test it**:
   - Open http://localhost:3000
   - You should see the landing page
   - Click "IMDB Sentiment Analyzer" card
   - Test the sentiment analysis

### Option 2: Deploy to Production

1. **Update Backend on Render**:
   - Push code to GitHub
   - Render will auto-deploy (or manually trigger)
   - New endpoint structure: `/api/nlp/predict` instead of `/predict`
   - Test: `https://your-app.onrender.com/api/nlp/health`

2. **Update Frontend API Call**:

   Edit `sentiment-frontend/src/pages/NLPSentiment/NLPSentiment.js` line 14:
   ```javascript
   const res = await fetch('https://your-render-app.onrender.com/api/nlp/predict', {
   ```

   Replace `your-render-app` with your actual Render app name.

3. **Update Vercel Deployment**:
   - Make sure Vercel still points to `sentiment-frontend` folder
   - OR rename `sentiment-frontend` to `frontend` and update Vercel config
   - Push to GitHub (Vercel will auto-deploy)

## Adding Your Next App

When you're ready to add a second app (e.g., another ML project):

1. **Create app folder**: `apps/my-new-app/`

2. **Create blueprint**: `apps/my-new-app/my_app_blueprint.py`
   ```python
   from flask import Blueprint, jsonify

   my_app_bp = Blueprint('my_app', __name__, url_prefix='/api/my-app')

   @my_app_bp.route('/endpoint')
   def endpoint():
       return jsonify({"message": "Hello from new app"})
   ```

3. **Register in main app.py**:
   ```python
   from apps.my_new_app.my_app_blueprint import my_app_bp
   app.register_blueprint(my_app_bp)
   ```

4. **Create React component**: `sentiment-frontend/src/pages/MyNewApp/MyNewApp.js`

5. **Add route**: In `sentiment-frontend/src/App.js`
   ```javascript
   <Route path="/my-new-app" element={<MyNewApp />} />
   ```

6. **Add card to home page**: In `sentiment-frontend/src/pages/Home.js`
   ```javascript
   {
     id: 'my-new-app',
     title: 'My New App',
     description: 'Description here',
     tech: ['Python', 'React', 'TensorFlow'],
     path: '/my-new-app',
     color: '#2196F3'
   }
   ```

## Files to Review

- **README.md** - Updated with new portfolio structure
- **DEPLOYMENT.md** - Complete deployment guide
- **apps/README.md** - Guide for adding new apps
- **apps/nlp-sentiment/README.md** - NLP app documentation

## Benefits

✅ **Resource Efficient**: Only one app runs at a time (lazy loading)
✅ **Easy to Expand**: Add new apps without touching existing ones
✅ **Professional**: Portfolio landing page showcases all your work
✅ **Maintainable**: Each app is isolated in its own folder
✅ **Single Deployment**: One backend, one frontend deployment

## Questions?

See `DEPLOYMENT.md` for detailed deployment instructions.
See `apps/README.md` for adding new applications.
See `README.md` for project overview.

---

**Ready to test?** Start with Option 1 above to test locally!
