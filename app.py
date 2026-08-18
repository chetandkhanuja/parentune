"""
Parentune Conversational Intelligence — NLP Project Prototype
IIM Amritsar — NLP Term IV

Pipeline:
  Module 1  — Topic classification         : TF-IDF + Logistic Regression, TRAINED OFFLINE by
                                              train_model.py and loaded here from models/topic_classifier.joblib
  Module 2  — Similar discussion retrieval : TF-IDF + cosine similarity (local, unsupervised, built at startup)
  Module 3  — Risk / warning flagging      : Gemini API (LLM call, no local training)
  Module 4a — Grounded answer generation   : Gemini API (LLM call, RAG-style — grounded in Module 2's
                                              retrieved discussion when one is found, general knowledge
                                              otherwise) — this is the actual response to the parent
  Module 4b — Discussion summarisation     : Gemini API (LLM call, no local training)
  Module 5  — Conversational interface     : Streamlit chat UI

NOTE: This app does not train anything itself. Run `python train_model.py` once
(and again whenever data/training_queries.csv changes) before starting this app.
"""

import os
import re
import time
import joblib
import concurrent.futures
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from google import genai

# --------------------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Parentune Conversational Intelligence", page_icon="👶", layout="centered")

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "models", "topic_classifier.joblib")
SIMILARITY_THRESHOLD = 0.15  # below this, we treat it as "no strongly similar discussion found"

# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------
@st.cache_data
def load_discussions():
    df = pd.read_csv(os.path.join(DATA_DIR, "discussions.csv"))
    df["full_text"] = df["title"] + ". " + df["comments"].str.replace("|||", " ", regex=False)
    return df


# --------------------------------------------------------------------------------------
# Module 1 — topic classifier (LOADED, not trained here — see train_model.py)
# --------------------------------------------------------------------------------------
@st.cache_resource
def load_classifier():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. "
            f"Run `python train_model.py` once before starting the app."
        )
    bundle = joblib.load(MODEL_PATH)
    return bundle["vectorizer"], bundle["clf"]


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
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"


def get_gemini_model_name():
    """Reads the model name from Streamlit secrets if set, so it can be changed
    without editing code — Google has been renaming/retiring free-tier models
    frequently. Falls back to DEFAULT_GEMINI_MODEL."""
    try:
        if "GEMINI_MODEL" in st.secrets:
            return st.secrets["GEMINI_MODEL"]
    except Exception:
        pass
    return DEFAULT_GEMINI_MODEL


GEMINI_MODEL = get_gemini_model_name()


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
# Module 4a — grounded answer generation via Gemini API
# --------------------------------------------------------------------------------------
def generate_answer(query, category, context_text, api_key):
    """Generates the actual reply to the parent's question.
    If context_text is provided (a similar past discussion's comments, surfaced by
    Module 2's TF-IDF retrieval), the answer is grounded in it — retrieval-augmented
    generation rather than the model answering from general knowledge alone.
    If context_text is None, no sufficiently similar discussion was found, so the
    model answers from general knowledge instead."""
    client = get_gemini_client(api_key)

    safety_rule = (
        "Never suggest a specific medicine, syrup, tablet, or dosage for a child. "
        "For anything that sounds medical, tell the parent to consult a pediatrician instead."
    )

    if context_text:
        prompt = (
            "You are a warm, practical parenting assistant answering a parent directly. "
            "Below is an excerpt from a community discussion related to their question — use it as "
            "supporting context ONLY if it is genuinely relevant to the question; if it isn't relevant, "
            "ignore it and answer from your own knowledge instead. Answer in 4-6 sentences, direct and "
            "specific. Do not mention that you were given context or reference 'the discussion' "
            f"explicitly — just answer naturally, as if this is your own advice. {safety_rule}\n\n"
            f"Parent's question (topic: {category}): {query}\n\n"
            f"Related community discussion excerpt:\n{context_text}"
        )
    else:
        prompt = (
            "You are a warm, practical parenting assistant answering a parent directly. "
            f"Answer in 4-6 sentences, direct and specific. {safety_rule}\n\n"
            f"Parent's question (topic: {category}): {query}"
        )

    response = call_gemini_with_retry(client, prompt)
    return response.text.strip()


# --------------------------------------------------------------------------------------
# Module 4b — summarisation via Gemini API
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
    vectorizer, clf = load_classifier()
    category, confidence = classify_query(query, vectorizer, clf)

    similar = retrieve_similar_discussions(query, top_n=3)
    top_match = similar.iloc[0]
    is_similar_found = top_match["similarity"] >= SIMILARITY_THRESHOLD

    summary, flagged, reason = None, False, None

    if is_similar_found:
        comments_text = top_match["comments"]
        # Fire all three Gemini calls at once instead of sequentially — this roughly
        # a third of the wait compared to running them one after another.
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            answer_future = executor.submit(generate_answer, query, category, comments_text, api_key)
            summary_future = executor.submit(summarise_discussion, comments_text, api_key)
            flag_future = executor.submit(flag_risky_language, comments_text, api_key)

            # The direct answer is the primary output — if it fails, that's the error
            # that surfaces, even if summary/flag happened to succeed.
            answer = answer_future.result()
            summary = summary_future.result()
            flagged, reason = flag_future.result()
    else:
        # No grounding discussion available — answer from general knowledge alone.
        answer = generate_answer(query, category, None, api_key)

    return {
        "answer": answer,
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

    # The direct answer leads — it's the actual response to what the parent asked.
    lines.append(result["answer"])

    lines.append(
        f"*(Classified as: {result['category']}, confidence: {result['confidence']*100:.0f}%)*"
    )

    if not result["is_similar_found"]:
        lines.append("I couldn't find a closely related past discussion for this query yet, "
                      "so the answer above is based on general knowledge rather than community threads.")
        return "\n\n".join(lines)

    lines.append("This is grounded in similar past discussions I found:")
    for _, row in result["similar"].iterrows():
        lines.append(f"- *{row['title']}* (similarity: {row['similarity']:.2f})")

    lines.append(f"**Summary of the most relevant discussion** — *{result['top_match']['title']}*:")
    lines.append(result["summary"])

    if result["flagged"]:
        lines.append(f"⚠️ **Possible medical recommendation flagged for moderator review.** {result['reason']}")
    else:
        lines.append("✅ No risky medical language detected in this discussion's comments.")

    return "\n\n".join(lines)


def main():
    st.title("👶 Parentune Conversational Intelligence")
    st.caption(
        "Academic NLP prototype — topic classification runs from a pre-trained model, "
        "similarity search runs locally; answer generation, summarisation, and "
        "risk-flagging are powered by the Gemini API."
    )

    # Fail fast with a clear message if the model hasn't been trained yet.
    try:
        load_classifier()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    # API key comes from Streamlit secrets (set in Streamlit Cloud's App settings -> Secrets).
    # No sidebar/UI input for it, so nothing sensitive is shown or asked for at demo time.
    api_key = st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else ""

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
            reply = ("Gemini API key isn't configured. Add GEMINI_API_KEY under this app's "
                      "Settings -> Secrets on Streamlit Cloud, then reboot the app.")
        else:
            with st.spinner("Classifying, retrieving similar discussions, and generating an answer..."):
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
