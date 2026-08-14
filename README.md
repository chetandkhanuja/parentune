# Parentune Conversational Intelligence — Chatbot Prototype

Academic NLP prototype for IIM Amritsar, NLP Term IV.

- **Module 1 — Topic classification**: TF-IDF + Logistic Regression, trained locally on a small sample dataset (`data/training_queries.csv`).
- **Module 2 — Similar discussion retrieval**: TF-IDF + cosine similarity over `data/discussions.csv`.
- **Module 3 — Risk / warning flagging**: Gemini API call (no training, no rules).
- **Module 4 — Discussion summarisation**: Gemini API call (no training).
- **Module 5 — Conversational interface**: Streamlit chat UI.

No model is trained beyond the small local classifier for Module 1 (as specified — the summarisation and
flagging steps use the Gemini API instead of a locally-run pretrained model or hand-written rules).

---

## 1. Get a free Gemini API key

1. Go to https://aistudio.google.com/apikey
2. Sign in with a Google account and click **Create API key**.
3. Copy the key — you'll paste it into the app (sidebar) or into Streamlit's secrets (for cloud deployment).

This is free (no credit card required) with a daily quota that's more than enough for a class demo.

---

## 2. Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Paste your Gemini API key into the sidebar text box when the app opens in your browser.

(Optional) Instead of pasting the key every time, copy `.streamlit/secrets.toml.example` to
`.streamlit/secrets.toml` and put your key there — the app will pick it up automatically.
This file is already in `.gitignore` so it won't be committed.

---

## 3. Deploy for free — Streamlit Community Cloud

This is the simplest free hosting option and is what your professor most likely expects.

1. **Push this folder to a GitHub repo** (public or private):
   ```bash
   cd parentune-bot
   git init
   git add .
   git commit -m "Parentune conversational intelligence prototype"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
   (`.streamlit/secrets.toml` will NOT be pushed — that's intentional, see `.gitignore`.)

2. Go to https://share.streamlit.io and sign in with GitHub.

3. Click **New app**, select your repo/branch, and set the main file path to `app.py`.

4. Before (or after) deploying, open **App settings → Secrets** in Streamlit Cloud and paste:
   ```toml
   GEMINI_API_KEY = "your-actual-key-here"
   ```
   Save. The app will restart and read the key automatically — nobody using your demo link
   will need to enter a key themselves.

5. Click **Deploy**. You'll get a public URL like `https://your-app-name.streamlit.app` you can
   share with your professor/group and put in your report.

**Note:** since your key lives in Streamlit's secrets (not in the sidebar), you can remove the sidebar
key input for the final cloud version if you want a cleaner demo — it's left in for now so it also
works for local testing without any setup.

---

## 4. Files

```
parentune-bot/
├── app.py                          # Streamlit app — all 5 pipeline modules
├── requirements.txt
├── data/
│   ├── training_queries.csv        # labelled sample queries for Module 1 classifier
│   └── discussions.csv             # sample discussions/comments for Modules 2–4
├── .streamlit/
│   └── secrets.toml.example        # copy to secrets.toml locally, or paste into Cloud secrets
└── .gitignore
```

## 5. Swapping in your real data

Replace `data/training_queries.csv` and `data/discussions.csv` with your actual labelled Reddit /
Stack Exchange sample once it's ready — the column names (`text`/`category` and
`discussion_id`/`category`/`title`/`comments`) just need to stay the same, everything else in
`app.py` will keep working.
