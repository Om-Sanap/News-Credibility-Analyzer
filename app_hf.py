# app_hf.py
import json
import streamlit as st
from transformers import pipeline, AutoTokenizer

# If you used the training script I provide (train_hf_from_true_false.py),
# the model directory and labels.json will be here:
MODEL_DIR = "bunpine/news-credibility-analyzer"

@st.cache_resource
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    clf = pipeline(
        "text-classification",
        model=MODEL_DIR,
        tokenizer=tok,
        device_map="auto"  # uses GPU if available; else CPU
    )
    with open(f"{MODEL_DIR}/labels.json", "r") as f:
        meta = json.load(f)
    id2label = {int(k): v for k, v in meta["id2label"].items()}
    return clf, tok, id2label

clf, tok, id2label = load_model()

st.set_page_config(page_title="Fake News Detector (HF)", page_icon="📰")
st.title("📰 Fake News Detector (Hugging Face)")
st.write("Paste an article or headline. The model predicts **real** vs **fake**.")

text = st.text_area("Article text", height=240, placeholder="Paste news text here...")

def classify_short(t):
    out = clf(t, truncation=True)[0]
    return out["label"], float(out["score"])

def classify_long(t, chunk_tokens=400):
    enc = tok(t, return_overflowing_tokens=True, truncation=True, max_length=chunk_tokens)
    preds = []
    for ids in enc["input_ids"]:
        chunk = tok.decode(ids, skip_special_tokens=True)
        preds.append(clf(chunk)[0])
    votes = {}
    sums = {}
    for p in preds:
        lab = p["label"]
        sc = float(p["score"])
        votes[lab] = votes.get(lab, 0) + 1
        sums[lab] = sums.get(lab, 0.0) + sc
    best = max(votes.items(), key=lambda kv: (kv[1], sums.get(kv[0], 0.0)))[0]
    conf = sums[best] / votes[best]
    return best, conf

col1, col2 = st.columns([3,1])
with col2:
    if st.button("Classify", use_container_width=True):
        if not text.strip():
            st.warning("Please paste some text.")
        else:
            n_tokens = len(tok(text)["input_ids"])
            if n_tokens <= 512:
                label, score = classify_short(text)
            else:
                label, score = classify_long(text)

            st.subheader("Prediction")
            st.write(f"**Label:** {label}")
            st.write(f"**Confidence:** {score:.2%}")
