# Streamlit Cloud Deployment Guide

## Prerequisites

- GitHub account
- Streamlit Community Cloud account (free at https://streamlit.io/cloud)
- Groq API key

## Step 1: Prepare Your Repository

### 1.1 Initialize Git (if not already done)
```bash
cd "/Users/mithil/Desktop/rag star"
git init
```

### 1.2 Add All Files
```bash
git add .
git commit -m "Initial commit: Enterprise RAG Analytics Engine"
```

### 1.3 Create GitHub Repository
1. Go to https://github.com/new
2. Name it: `enterprise-rag-analytics` (or your preferred name)
3. **DO NOT** initialize with README, .gitignore, or license (we already have these)
4. Click "Create repository"

### 1.4 Push to GitHub
```bash
git remote add origin https://github.com/YOUR_USERNAME/enterprise-rag-analytics.git
git branch -M main
git push -u origin main
```

## Step 2: Deploy on Streamlit Cloud

### 2.1 Connect Repository
1. Go to https://share.streamlit.io/
2. Click "New app"
3. Select your GitHub repository: `YOUR_USERNAME/enterprise-rag-analytics`
4. Set **Main file path**: `app_enterprise.py`
5. Click "Deploy"

### 2.2 Configure Secrets (CRITICAL)
1. After deployment starts, click "Advanced settings" or go to app settings
2. Navigate to "Secrets" section
3. Paste your Groq API key:
   ```toml
   GROQ_API_KEY = "gsk_your_actual_key_here"
   ```
4. Click "Save"

### 2.3 Wait for Deployment
- First deployment takes 2-5 minutes
- App will automatically rebuild when you push changes to GitHub

## Step 3: Verify Deployment

1. Once deployed, test with these queries:
   - "Total industry premium FY25?"
   - "Top 5 companies by premium"
   - "Health segment YoY growth?"

2. Check that all features work:
   - Document count shows 52
   - Queries return proper metrics
   - Sources are properly cited

## Troubleshooting

### Issue: "GROQ_API_KEY not found"
**Fix:** Add API key in Streamlit Cloud secrets (Step 2.2)

### Issue: "Module not found"
**Fix:** Verify `requirements.txt` has all dependencies

### Issue: "ChromaDB initialization failed"
**Fix:** ChromaDB will be recreated on first run - this is normal

### Issue: Rate limit errors
**Fix:** Groq free tier has daily limits. System falls back to template mode automatically.

## Security Checklist

- ✅ `.env` file is in `.gitignore`
- ✅ No API keys committed to GitHub
- ✅ Secrets configured in Streamlit Cloud
- ✅ `.gitignore` prevents accidental key exposure

## Post-Deployment

### Updating Your App
```bash
# Make changes to your code
git add .
git commit -m "Description of changes"
git push
```

Streamlit Cloud will automatically redeploy when you push.

### Monitoring
- Check app logs in Streamlit Cloud dashboard
- Monitor Groq API usage at https://console.groq.com

## App URL
After deployment, your app will be at:
```
https://YOUR_USERNAME-enterprise-rag-analytics-app-enterprise-xxxxx.streamlit.app
```

Share this URL with stakeholders for access to your Enterprise RAG Analytics Engine!
