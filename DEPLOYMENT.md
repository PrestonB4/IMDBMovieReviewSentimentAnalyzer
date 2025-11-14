# Deployment Guide

## Backend Deployment (Render.com)

### First Time Setup

1. Push your code to GitHub
2. Go to [Render.com](https://render.com/) and create a new Web Service
3. Connect your GitHub repository
4. Configure the service:
   - **Name**: `portfolio-backend` (or your preferred name)
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Branch**: `main`

### Important Files

- `requirements.txt` - Must include all dependencies
- `app.py` - Main application file
- `apps/` - All application blueprints and models

### Environment Variables (if needed)
No environment variables required for basic setup.

### After Deployment

Your backend will be available at: `https://your-app-name.onrender.com`

Test it:
```bash
curl https://your-app-name.onrender.com/
curl https://your-app-name.onrender.com/api/nlp/health
```

---

## Frontend Deployment (Vercel)

### Before Deploying

**IMPORTANT**: Update the API endpoint in your app components.

In `frontend/src/pages/NLPSentiment/NLPSentiment.js`, update line 14:

```javascript
// For production (Render backend)
const res = await fetch('https://YOUR-RENDER-APP.onrender.com/api/nlp/predict', {

// For local development
const res = await fetch('http://localhost:5000/api/nlp/predict', {
```

### First Time Setup

1. Go to [Vercel](https://vercel.com/)
2. Import your GitHub repository
3. Configure the project:
   - **Framework Preset**: Create React App
   - **Root Directory**: `frontend` (or `sentiment-frontend`)
   - **Build Command**: `npm run build`
   - **Output Directory**: `build`

4. Deploy!

### After Deployment

Your frontend will be available at: `https://your-app.vercel.app`

---

## Local Development

### For local testing with both frontend and backend:

1. In `frontend/src/pages/NLPSentiment/NLPSentiment.js`, use:
   ```javascript
   const res = await fetch('http://localhost:5000/api/nlp/predict', {
   ```

2. Run backend:
   ```bash
   python app.py
   ```

3. Run frontend (in separate terminal):
   ```bash
   cd frontend
   npm start
   ```

### For frontend testing with production backend:

1. In `frontend/src/pages/NLPSentiment/NLPSentiment.js`, use:
   ```javascript
   const res = await fetch('https://your-render-app.onrender.com/api/nlp/predict', {
   ```

2. Run frontend:
   ```bash
   cd frontend
   npm start
   ```

---

## Adding New Apps to Deployment

When you add a new app:

1. **Backend**:
   - Add the blueprint to `app.py`
   - The new routes will automatically be available when you redeploy

2. **Frontend**:
   - Add the new route to `App.js`
   - Add the app card to `Home.js`
   - Update the component with the correct API endpoint
   - Vercel will auto-deploy on git push (if enabled)

---

## Troubleshooting

### Backend Issues

**Problem**: "Module not found" error
- **Solution**: Make sure all dependencies are in `requirements.txt`

**Problem**: Cold start takes 30-60 seconds
- **Solution**: This is normal on Render free tier. First request after inactivity will be slow.

**Problem**: "Import error" for blueprints
- **Solution**: Verify `__init__.py` files exist in `apps/` and app folders

### Frontend Issues

**Problem**: "Failed to fetch" or CORS error
- **Solution**: Verify backend URL is correct and CORS is enabled in Flask

**Problem**: 404 on refresh
- **Solution**: Add a `vercel.json` file with rewrites (see below)

### Vercel 404 Fix

Create `frontend/vercel.json`:
```json
{
  "rewrites": [
    { "source": "/(.*)", "destination": "/" }
  ]
}
```

This ensures React Router handles all routes.
