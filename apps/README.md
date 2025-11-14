# Apps Directory

This directory contains all individual applications in the portfolio.

## Structure

Each app should be in its own subdirectory with the following structure:

```
apps/
  app-name/
    __init__.py
    app_blueprint.py      # Flask Blueprint
    models/              # ML models or data files
    static/              # Static assets (if any)
    README.md            # App-specific documentation
```

## Creating a New App

1. Create a new folder: `apps/your-app-name/`
2. Create `__init__.py` to make it a Python package
3. Create `your_app_blueprint.py` with Flask Blueprint:

```python
from flask import Blueprint

your_app_bp = Blueprint('your_app', __name__, url_prefix='/api/your-app')

# Add routes here
@your_app_bp.route('/endpoint')
def endpoint():
    return {"message": "Your app endpoint"}
```

4. Register the blueprint in the main `app.py`:

```python
from apps.your_app.your_app_blueprint import your_app_bp
app.register_blueprint(your_app_bp)
```

5. Create the frontend component in `frontend/src/pages/YourApp/`

6. Add the route in `frontend/src/App.js`:

```javascript
<Route path="/your-app" element={<YourApp />} />
```

7. Add the app card to `frontend/src/pages/Home.js`

## Current Apps

- **nlp-sentiment**: IMDB Movie Review Sentiment Analyzer
