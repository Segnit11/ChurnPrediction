# Deployment Guide

ChurnGuard deploys as two pieces:

- **Backend** (Flask + ML models) → **Hugging Face Spaces** (free, Docker SDK, 16 GB RAM — needed for the ~629 MB models).
- **Frontend** (React) → **Vercel** (free).

> Deploy the **backend first** so you have its URL, then point the frontend at it.

All the config is already in this repo: `Dockerfile`, `.dockerignore`,
`requirements-deploy.txt`, the HF metadata block at the top of `README.md`, and
`frontend/vercel.json`. You only need to do the account/auth steps below.

---

## 1. Backend → Hugging Face Spaces

The two large models (`churn_model_stacking.pkl` 417 MB, `rf_model.pkl` 209 MB)
are **not** in git — `app.py` downloads them from Google Drive on first boot
(takes ~1–2 min on the first start; the Space shows "Starting" until ready).

### One-time setup
1. Create a free account at https://huggingface.co and a token at
   https://huggingface.co/settings/tokens (role: **Write**).
2. Create a new Space: https://huggingface.co/new-space
   - **SDK:** Docker → *Blank*
   - **Hardware:** CPU basic (free)
   - Name it e.g. `churnguard-api`.

### Push the code (uses Git LFS for the 12 MB dataset)
```bash
# from the repo root
git lfs install
# Add the Space as a remote (replace <user>)
git remote add space https://huggingface.co/spaces/<user>/churnguard-api
# Track files >10 MB so HF accepts them
git lfs track "data/*.csv"
git add .gitattributes
git commit -m "Track dataset with LFS for HF" || true
# Push current branch to the Space's main
git push space HEAD:main
```
When prompted for a password, paste your **HF write token**.

> Alternatively (no LFS fuss), upload via Python — it handles large files
> automatically:
> ```bash
> pip install huggingface_hub
> huggingface-cli login   # paste write token
> python -c "from huggingface_hub import HfApi; HfApi().upload_folder(repo_id='<user>/churnguard-api', repo_type='space', folder_path='.', ignore_patterns=['venv/*','.venv/*','frontend/*','notebooks/*','*.pkl' if False else '','__pycache__/*'])"
> ```

### Verify
Once the Space says **Running**, your API is at:
```
https://<user>-churnguard-api.hf.space
```
Test it: open `https://<user>-churnguard-api.hf.space/health` → should return JSON
with `"status":"ok"` and 4 models. **Copy this base URL** for the next step.

---

## 2. Frontend → Vercel

1. Go to https://vercel.com → **Add New → Project** → import
   `Segnit11/ChurnPrediction` from GitHub.
2. **Root Directory:** set to `frontend`.
3. Framework preset auto-detects **Create React App** (build command
   `npm run build`, output `build`). `vercel.json` already pins this and adds SPA
   rewrites.
4. **Environment Variables** → add:
   | Name | Value |
   |------|-------|
   | `REACT_APP_API_URL` | `https://<user>-churnguard-api.hf.space` (no trailing slash) |
   > CRA inlines env vars at **build time**, so this must be set before/at deploy.
5. **Deploy.** You'll get a URL like `https://churn-prediction-xxxx.vercel.app`.

That's it — the deployed frontend will call the Hugging Face backend for live
predictions, analytics, and the customer list. CORS is already open on the API.

---

## Notes & gotchas
- **First backend request is slow** (model download + load on cold start). HF
  free Spaces also sleep after inactivity and cold-start again on next hit.
- **Gemini explanations:** optional. Add a `GEMINI_API_KEY` secret in the Space
  settings to switch from rule-based to AI-generated explanations/emails.
- **Re-deploying the frontend** after changing `REACT_APP_API_URL` requires a
  fresh build (Vercel does this automatically on env change + redeploy).
- **Local dev** is unchanged: backend `python app.py` (port 5001), frontend
  `npm start` (defaults to `http://localhost:5001`).
