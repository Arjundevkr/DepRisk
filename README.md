# DepRisk — Depression Risk Predictor

A web-based depression risk prediction tool based on screen time and sleep quality data.

## Live Demo

Visit: **https://arjundevkr.github.io/deprisk-app/**

## Files

- `index.html` — Main webpage
- `style.css` — Dark theme styling
- `app.js` — Logistic regression prediction model

## Deployment to GitHub Pages

```bash
# 1. Go to the folder
cd gh-pages-deploy

# 2. Initialize git (if not done)
git init

# 3. Commit files
git add -A
git commit -m "Initial DepRisk app"

# 4. Create GitHub repo (on web.github.com), then:
git remote add origin https://github.com/YOURUSERNAME/deprisk-app.git
git push -u origin main

# 5. Enable GitHub Pages:
#    Repo → Settings → Pages → Source: main → Save
```

## About

Built with:
- HTML5 + CSS3 (dark theme)
- Chart.js for visualizations
- Vanilla JavaScript logistic regression model

## Credits

Dataset: Screen Time, Sleep & Adolescent Mental Health Study