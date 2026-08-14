"""
Parentune Conversational Intelligence — NLP Project Prototype
IIM Amritsar — NLP Term IV
 
Pipeline:
  Module 1 — Topic classification        : TF-IDF + Logistic Regression (local, trained on sample data)
  Module 2 — Similar discussion retrieval: TF-IDF + cosine similarity (local)
  Module 3 — Risk / warning flagging     : Gemini API (LLM call, no local training)
  Module 4 — Discussion summarisation    : Gemini API (LLM call, no local training)
  Module 5 — Conversational interface    : Streamlit chat UI, template-driven output
"""
 
import os
import re
import time
import concurrent.futures
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
 
from google import genai
 
# --------------------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Parentune Conversational Intelligence", page_icon="👶", layout="centered")
 
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SIMILARITY_THRESHOLD = 0.15  # below this, we tell the user no strongly similar discussion was found
 
# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------
@st.cache_data
def load_training_data():
    return pd.read_csv(os.path.join(DATA_DIR, "training_queries.csv"))
 
 
@st.cache_data
def load_discussions():
    df = pd.read_csv(os.path.join(DATA_DIR, "discussions.csv"))
    df["full_text"] = df["title"] + ". " + df["comments"].str.replace("|||", " ", regex=False)
    return df
 
 
# --------------------------------------------------------------------------------------
# Module 1 — topic classifier (local, no external API)
# --------------------------------------------------------------------------------------
@st.cache_resource
def train_classifier():
    train_df = load_training_data()
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(train_df["text"])
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, train_df["category"])
    return vectorizer, clf
 
 
def classify_query(query, vectorizer, clf):
    X = vectorizer.transform([query])
    pred = clf.predict(X)[0]
    proba = clf.predict_proba(X).max()
    return pred, proba
 
 
# --------------------------------------------------------------------------------------
# Module 2 — similarity / duplicate-discussion retrieval (local, no external API)
# --------------------------------------------------------------------------------------
@st.cache_resource
def build_similarity_index():
    discussions_df = load_discussions()
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(discussions_df["full_text"])
    return vectorizer, X
 
 
def retrieve_similar_discussions(query, top_n=3):
    vectorizer, X = build_similarity_index()
    discussions_df = load_discussions()
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, X).flatten()
    top_idx = sims.argsort()[::-1][:top_n]
    results = discussions_df.iloc[top_idx].copy()
    results["similarity"] = sims[top_idx]
    return results
 
 
# --------------------------------------------------------------------------------------
# Gemini API setup
# --------------------------------------------------------------------------------------
GEMINI_MODEL = "gemini-flash-latest"
 
 
def get_gemini_client(api_key):
    return genai.Client(api_key=api_key)
 
 
def call_gemini_with_retry(client, prompt, max_attempts=4, base_delay=2):
    """Calls the Gemini API, retrying with backoff if the model is temporarily
    overloaded (503 UNAVAILABLE) or rate-limited (429). Other errors raise immediately."""
    last_error = None
    for attempt in range(max_attempts):
        try:
            return client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        except Exception as e:
            last_error = e
            message = str(e)
            is_retryable = "UNAVAILABLE" in message or "503" in message or "429" in message
            if not is_retryable or attempt == max_attempts - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))  # 2s, 4s, 8s...
    raise last_error
 
 
# --------------------------------------------------------------------------------------
# Module 4 — summarisation via Gemini API
# --------------------------------------------------------------------------------------
def summarise_discussion(comments_text, api_key):
    client = get_gemini_client(api_key)
    prompt = (
        "Summarise the following parenting discussion comments in 3-4 sentences. "
        "Focus only on the practical advice and takeaways parents shared. "
        "Do not add any information that is not in the comments.\n\n"
        f"Comments:\n{comments_text}"
    )
    response = call_gemini_with_retry(client, prompt)
    return response.text.strip()
 
 
# --------------------------------------------------------------------------------------
# Module 3 — risk / warning flagging via Gemini API
# --------------------------------------------------------------------------------------
def flag_risky_language(comments_text, api_key):
    client = get_gemini_client(api_key)
    prompt = (
        "You are a moderation assistant for a parenting community platform. "
        "Read the comments below and check ONLY for informal medical recommendations "
        "aimed at a child — e.g. suggesting a specific medicine, syrup, tablet, or dosage "
        "(such as 'give him 5ml of X' or 'we just gave her Y tablets'). "
        "This is not a diagnosis and not medical advice — you are only flagging text for "
        "a human moderator to review.\n\n"
        f"Comments:\n{comments_text}\n\n"
        "Respond in exactly this format, nothing else:\n"
        "FLAG: YES or NO\n"
        "REASON: one short sentence"
    )
    response = call_gemini_with_retry(client, prompt)
    text = response.text.strip()
 
    flag_match = re.search(r"FLAG:\s*(YES|NO)", text, re.IGNORECASE)
    reason_match = re.search(r"REASON:\s*(.+)", text, re.IGNORECASE)
 
    flagged = bool(flag_match and flag_match.group(1).upper() == "YES")
    reason = reason_match.group(1).strip() if reason_match else text
    return flagged, reason
 
 
# --------------------------------------------------------------------------------------
# Module 5 — conversational interface
# --------------------------------------------------------------------------------------
def run_pipeline(query, api_key):
    vectorizer, clf = train_classifier()
    category, confidence = classify_query(query, vectorizer, clf)
 
    similar = retrieve_similar_discussions(query, top_n=3)
    top_match = similar.iloc[0]
    is_similar_found = top_match["similarity"] >= SIMILARITY_THRESHOLD
 
    summary, flagged, reason = None, False, None
    if is_similar_found:
        comments_text = top_match["comments"]
        # Fire both Gemini calls at once instead of waiting for one to finish before
        # starting the other — this roughly halves the wait compared to sequential calls.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            summary_future = executor.submit(summarise_discussion, comments_text, api_key)
            flag_future = executor.submit(flag_risky_language, comments_text, api_key)
 
            # The summary is the primary output — if it fails, surface that error and
            # don't bother using the flag result even if it happened to succeed.
            summary = summary_future.result()
            flagged, reason = flag_future.result()
 
    return {
        "category": category,
        "confidence": confidence,
        "similar": similar,
        "is_similar_found": is_similar_found,
        "top_match": top_match,
        "summary": summary,
        "flagged": flagged,
        "reason": reason,
    }
 
 
def format_bot_reply(result):
    lines = []
    lines.append(
        f"**I categorised this as: {result['category']}** "
        f"(confidence: {result['confidence']*100:.0f}%)"
    )
 
    if not result["is_similar_found"]:
        lines.append("I couldn't find a closely related past discussion for this query yet.")
        return "\n\n".join(lines)
 
    lines.append("I found some similar past discussions:")
    for _, row in result["similar"].iterrows():
        lines.append(f"- *{row['title']}* (similarity: {row['similarity']:.2f})")
 
    lines.append(f"**Summary of the most relevant discussion** — *{result['top_match']['title']}*:")
    lines.append(result["summary"])
 
    if result["flagged"]:
        lines.append(f"⚠️ **Possible medical recommendation flagged for moderator review.** {result['reason']}")
    else:
        lines.append("✅ No risky medical language detected in this discussion's comments.")
 
    return "\n\n".join(lines)
 
 
# --------------------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------------------
def main():
    st.title("👶 Parentune Conversational Intelligence")
    st.caption(
        "Academic NLP prototype — topic classification & similarity search run locally; "
        "summarisation and risk-flagging are powered by the Gemini API (no model training)."
    )
 
    with st.sidebar:
        st.header("Setup")
        default_key = st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else ""
        api_key = st.text_input(
            "Gemini API key",
            value=default_key,
            type="password",
            help="Get a free key at https://aistudio.google.com/apikey. "
                 "On Streamlit Cloud, set this as a secret instead (see README).",
        )
        st.markdown("---")
        st.markdown(
            "**Pipeline:**\n"
            "1. Topic classification (local ML)\n"
            "2. Similar discussion retrieval (local TF-IDF)\n"
            "3. Risk flagging (Gemini API)\n"
            "4. Summarisation (Gemini API)\n"
            "5. Conversational output (this chat)"
        )
 
    if "history" not in st.session_state:
        st.session_state.history = []
 
    for role, content in st.session_state.history:
        with st.chat_message(role):
            st.markdown(content)
 
    query = st.chat_input("Ask a parenting question, e.g. 'my toddler won't eat vegetables'")
 
    if query:
        st.session_state.history.append(("user", query))
        with st.chat_message("user"):
            st.markdown(query)
 
        if not api_key:
            reply = "Please enter a Gemini API key in the sidebar to continue (it's free — see the link there)."
        else:
            with st.spinner("Classifying, retrieving similar discussions, and calling the API..."):
                try:
                    result = run_pipeline(query, api_key)
                    reply = format_bot_reply(result)
                except Exception as e:
                    message = str(e)
                    if "UNAVAILABLE" in message or "503" in message:
                        reply = ("Google's Gemini servers are overloaded right now, even after "
                                 "retrying a few times. This is temporary on Google's side — please "
                                 "try again in a minute.")
                    else:
                        reply = f"Something went wrong calling the Gemini API: {e}"
 
        st.session_state.history.append(("assistant", reply))
        with st.chat_message("assistant"):
            st.markdown(reply)
 
 
if __name__ == "__main__":
    main()
 
