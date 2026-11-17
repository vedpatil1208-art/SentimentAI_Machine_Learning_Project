import re
import joblib
import streamlit as st
import nltk
import pandas as pd
import plotly.express as px
import requests
from urllib.parse import urlparse

st.set_page_config(page_title="SentimentAI", page_icon="💬", layout="centered")

# Styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #100d20, #211447);
    color: white;
}
.block-container {
    max-width: 850px;
    padding-top: 2rem;
}
section[data-testid="stSidebar"] {
    display: none;
}
div.stButton > button {
    background: #7138c8;
    color: white;
    border-radius: 10px;
    min-height: 45px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Load trained model and vectorizer
@st.cache_resource
def load_models():
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords

    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    stop_words = set(stopwords.words("english"))
    stop_words -= {"not", "no", "nor", "never"}

    return model, vectorizer, stop_words

try:
    model, vectorizer, stop_words = load_models()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()

LABELS = {-1: "Negative", 0: "Neutral", 1: "Positive"}
COLORS = {
    "Positive": "#34d399",
    "Neutral": "#fbbf24",
    "Negative": "#fb7185"
}

# Same preprocessing as training notebook
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return " ".join(
        word for word in text.split()
        if word not in stop_words
    )

# Extract public Reddit post and comments
def extract_reddit(url):
    url = url.split("?")[0].rstrip("/")
    if not url.endswith(".json"):
        url += ".json"

    response = requests.get(
        url,
        headers={"User-Agent": "SentimentAI/1.0"},
        timeout=25
    )
    response.raise_for_status()
    data = response.json()

    post = data[0]["data"]["children"][0]["data"]
    texts = []

    if post.get("title"):
        texts.append(post["title"])
    if post.get("selftext"):
        texts.append(post["selftext"])

    def collect_comments(children):
        for child in children:
            item = child.get("data", {})
            body = item.get("body", "")

            if body and body not in ("[deleted]", "[removed]"):
                texts.append(body)

            replies = item.get("replies")
            if isinstance(replies, dict):
                collect_comments(
                    replies.get("data", {}).get("children", [])
                )

    if len(data) > 1:
        collect_comments(data[1]["data"]["children"])

    if not texts:
        raise ValueError("No public post or comments found.")

    return texts

# Analyze sentiment
def analyze(texts, source):
    cleaned = [preprocess_text(t) for t in texts]

    valid = [
        (original, processed)
        for original, processed in zip(texts, cleaned)
        if processed
    ]

    if not valid:
        raise ValueError("No meaningful English text found.")

    originals, processed = zip(*valid)
    vectors = vectorizer.transform(list(processed))
    predictions = model.predict(vectors)

    results = [LABELS[int(p)] for p in predictions]

    counts = pd.Series(results).value_counts().reindex(
        ["Positive", "Neutral", "Negative"], fill_value=0
    )

    dominant = counts.idxmax()

    st.markdown("---")
    st.subheader("Analysis Result")

    st.markdown(
        f"<h1 style='color:{COLORS[dominant]}'>{dominant}</h1>",
        unsafe_allow_html=True
    )
    st.caption(f"Source: {source} | Analyzed items: {len(results)}")

    # Pie chart
    if len(results) > 1:
        chart = pd.DataFrame({
            "Sentiment": counts.index,
            "Count": counts.values
        })
        fig = px.pie(
            chart,
            names="Sentiment",
            values="Count",
            hole=0.45,
            color="Sentiment",
            color_discrete_map=COLORS,
            title="Sentiment Distribution"
        )
        note = "Distribution of sentiment across retrieved posts/comments."

    elif hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vectors)[0]
        chart = pd.DataFrame({
            "Sentiment": [LABELS[int(c)] for c in model.classes_],
            "Probability": probabilities * 100
        })
        fig = px.pie(
            chart,
            names="Sentiment",
            values="Probability",
            hole=0.45,
            color="Sentiment",
            color_discrete_map=COLORS,
            title="Estimated Sentiment Probabilities"
        )
        note = "Model-estimated probabilities for this single post, not audience percentages."

    else:
        fig = px.pie(
            names=[dominant],
            values=[1],
            title="Predicted Sentiment"
        )
        note = "Predicted class for this single post."

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(note)

    # Simple review
    st.subheader("Simple Review")
    percentage = counts[dominant] / len(results) * 100

    if dominant == "Positive":
        review = "The analyzed content is mostly positive."
    elif dominant == "Negative":
        review = "The analyzed content is mostly negative."
    else:
        review = "The analyzed content is mostly neutral."

    st.write(
        f"{review} {percentage:.1f}% of analyzed text items "
        f"were classified as {dominant.lower()}."
    )

    with st.expander("View analyzed text"):
        for i, (text, sentiment) in enumerate(
            zip(originals, results), 1
        ):
            st.write(f"{i}. **{sentiment}:** {text[:500]}")

# Main application
st.title("💬 SentimentAI")
st.write("Analyze social media sentiment using Machine Learning.")

url = st.text_input(
    "Paste a social media URL",
    placeholder="Paste a public Reddit post URL..."
)

pasted_text = st.text_area(
    "Or paste post text/comments",
    height=100
)

st.caption(
    "Automatic URL extraction is supported for public Reddit posts. "
    "Other platforms may require API access or pasted text."
)

if st.button("Analyze Sentiment", type="primary", use_container_width=True):
    try:
        if url.strip():
            host = urlparse(url).netloc.lower().replace("www.", "")

            if "reddit.com" in host or host == "redd.it":
                with st.spinner("Fetching Reddit post and comments..."):
                    texts = extract_reddit(url)
                analyze(texts, "Reddit")

            elif pasted_text.strip():
                analyze([pasted_text.strip()], "Pasted text")

            else:
                st.warning(
                    "Automatic extraction for this platform is not configured. "
                    "Paste the post text or comments instead."
                )

        elif pasted_text.strip():
            analyze([pasted_text.strip()], "Pasted text")

        else:
            st.warning("Please enter a URL or some text.")

    except requests.RequestException as e:
        st.error(f"Could not retrieve the post: {e}")
    except (ValueError, KeyError, IndexError) as e:
        st.warning(str(e))

st.markdown("---")
st.caption("SentimentAI | Academic Machine Learning Project")
