# app_hf_fancy.py
import json, math, io
import numpy as np
import pandas as pd
import streamlit as st
from transformers import pipeline, AutoTokenizer

st.info(
    "This model identifies patterns associated with real and fake news datasets. "
    "It does not independently verify factual accuracy."
)
# ====== CONFIG ======
MODEL_DIR = "bunpine/news-credibility-analyzer"  # path created by your training script
APP_TITLE = "📰 Fake News Detector (Hugging Face • Pro UI)"

# ====== LOAD MODEL ======
# --- Sidebar model selector (safe mapping) ---
from pathlib import Path
from transformers import AutoTokenizer, pipeline
import streamlit as st

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📰",
    layout="wide"
)

MODEL_OPTIONS = {
    "Your fine-tuned (local)": "bunpine/news-credibility-analyzer"
}

selected_label = st.sidebar.selectbox("Choose model", list(MODEL_OPTIONS.keys()))
model_id = MODEL_OPTIONS[selected_label]  # <- real id/path without spaces

@st.cache_resource
def load_selected_model(model_id: str):
    is_local = Path(model_id).exists()
    if is_local:
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        clf = pipeline("text-classification", model=model_id, tokenizer=tok, device_map="auto")
    else:
        tok = AutoTokenizer.from_pretrained(model_id)
        clf = pipeline("text-classification", model=model_id, tokenizer=tok, device_map="auto")
    return clf, tok
st.write("DEBUG model_id =", model_id)
clf, tok = load_selected_model(model_id)
if selected_label.startswith("Your") and not Path("bunpine/news-credibility-analyzer").exists():
    st.warning("⚠️ Local fine-tuned model folder 'bunpine/news-credibility-analyzer' not found. "
               "Please train your model first or select a Hugging Face model from the sidebar.")
@st.cache_resource
def load_baseline_model():
    # baseline for comparison; loads once
    base_clf, base_tok = load_selected_model("distilroberta-base")
    return base_clf, base_tok


# ====== STYLES ======
st.set_page_config(page_title=APP_TITLE, page_icon="📰", layout="wide")
st.markdown("""
<style>
.badge {
  display:inline-block; padding:6px 10px; border-radius:999px; font-weight:600; font-size:0.9rem;
  background:#E5F4EA; color:#14532d; border:1px solid #16a34a22;
}
.badge.fake { background:#FDECEC; color:#7F1D1D; border-color:#ef444422; }
.badge.real { background:#EAF2FF; color:#1E3A8A; border-color:#3b82f622; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
.small { font-size: 0.9rem; opacity: 0.8; }
.token { padding:2px 4px; border-radius:6px; margin-right:2px; }
.section { border:1px solid #0001; border-radius:14px; padding:14px; }
</style>
""", unsafe_allow_html=True)

st.title(APP_TITLE)
st.write("Paste text, upload CSV, and show explanations & eval metrics. Built on a fine-tuned Transformer.")

# ====== HELPERS ======
def predict_short(text:str):
    out = clf(text, truncation=True)[0]
    label, score = out["label"], float(out["score"])
    return label, score

def predict_long(text:str, chunk_tokens=400):
    enc = tok(text, return_overflowing_tokens=True, truncation=True, max_length=chunk_tokens)
    input_ids = enc["input_ids"]
    preds = []
    for ids in input_ids:
        chunk = tok.decode(ids, skip_special_tokens=True)
        preds.append(clf(chunk)[0])
    votes, sums = {}, {}
    for p in preds:
        lab, sc = p["label"], float(p["score"])
        votes[lab] = votes.get(lab, 0) + 1
        sums[lab]  = sums.get(lab, 0.0) + sc
    best = max(votes.items(), key=lambda kv: (kv[1], sums.get(kv[0],0.0)))[0]
    conf = sums[best] / votes[best]
    # Also return per-label vote table
    vote_df = pd.DataFrame(
        [{"label": k, "votes": v, "avg_conf": round(sums[k]/v, 4)} for k,v in votes.items()]
    ).sort_values(["votes","avg_conf"], ascending=False)
    return best, conf, vote_df, preds

def confidence_meter(score: float):
    # Pretty progress + numeric
    st.progress(score)
    st.caption(f"Confidence: **{score:.1%}**")

def label_badge(label: str):
    css = "fake" if label.lower().startswith("fake") else "real"
    st.markdown(f'<span class="badge {css}">{label.upper()}</span>', unsafe_allow_html=True)

def split_sentences(text:str):
    # simple sentence splitter to avoid extra deps
    parts = [p.strip() for p in
             (text.replace("\n"," ").replace("?", ".").replace("!", ".").split("."))]
    return [p for p in parts if p]

def sentence_explain(text:str):
    sents = split_sentences(text)
    rows = []
    for s in sents:
        if not s.strip(): continue
        lab, sc = predict_short(s)
        prob_fake = sc if lab.lower()=="fake" else 1.0 - sc
        rows.append({"sentence": s, "label": lab, "prob_fake": prob_fake, "conf": sc})
    if not rows:
        return pd.DataFrame(columns=["sentence","label","prob_fake","conf"])
    df = pd.DataFrame(rows)
    return df

def color_sentence_html(sent, prob_fake):
    # green -> real (0), red -> fake (1)
    # map prob to color
    r = int(255 * prob_fake)
    g = int(255 * (1 - prob_fake))
    b = 200
    color = f"rgba({r},{g},{b},0.25)"
    return f'<span class="token" style="background:{color}">{sent}</span>'

# ====== SIDEBAR ======
with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Decision threshold for 'FAKE'", 0.2, 0.8, 0.5, 0.05,
                          help="If model confidence for FAKE ≥ threshold, we call it FAKE.")
    chunk_tokens = st.slider("Chunk size for long articles (tokens)", 200, 512, 400, 50)
    st.caption("Tip: Increase chunk size for speed, decrease for more granular votes.")

# ====== TABS ======
tab1, tab2, tab3, tab4 = st.tabs(["📝 Single Text", "📦 Batch & Export", "🧠 Explain", "📊 Evaluate"])

# --- TAB 1: Single Text ---
with tab1:
    colA, colB = st.columns([3,2])
    with colA:
        text = st.text_area("Article / Headline", height=220,
                            placeholder="Paste news article text here...")
        run = st.button("Classify", type="primary")
    with colB:
        st.markdown("#### Result")

        if run and text.strip():
            # choose path
            n_tokens = len(tok(text)["input_ids"])
            if n_tokens <= 512:
                label, score = predict_short(text)
                # apply threshold only when label is FAKE; otherwise invert check
                if label.lower()=="fake" and score < threshold:
                    label = "real"
                    score = 1.0 - score
            else:
                label, score, votes_df, raw_preds = predict_long(text, chunk_tokens=chunk_tokens)

            # --- Optional: show baseline comparison ---
            base_clf, base_tok = load_baseline_model()
            base_out = base_clf(text, truncation=True)[0]
            base_label, base_score = base_out["label"], float(base_out["score"])

            st.markdown("##### Baseline Comparison (distilroberta-base)")
            colX, colY = st.columns(2)
            with colX:
                st.write("**Selected Model:**")
                label_badge(label); confidence_meter(score)
            with colY:
                st.write("**Baseline Model:**")
                label_badge(base_label); confidence_meter(base_score)

            label_badge(label)
            confidence_meter(score)

            if n_tokens > 512:
                st.markdown("##### Chunk Votes")
                st.dataframe(votes_df, use_container_width=True)
                st.bar_chart(votes_df.set_index("label")["votes"])

    with st.expander("Advanced (show tokens)"):
        st.write("Tokenization preview (first 50 tokens):")
        if text.strip():
            ids = tok(text)["input_ids"][:50]
            st.code(ids, language="json")

# --- TAB 2: Batch & Export ---
with tab2:
    st.write("Upload a CSV with a **text** column. Get predictions + download the results.")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        # infer text column
        text_col = "text" if "text" in df.columns else df.columns[0]
        preds, confs = [], []

        with st.spinner("Predicting..."):
            for t in df[text_col].fillna(""):
                if not t.strip():
                    preds.append("")
                    confs.append(0.0)
                    continue
                n_tokens = len(tok(t)["input_ids"])
                if n_tokens <= 512:
                    lab, sc = predict_short(t)
                    if lab.lower()=="fake" and sc < threshold:
                        lab, sc = "real", 1.0 - sc
                else:
                    lab, sc, _, _ = predict_long(t, chunk_tokens=chunk_tokens)
                preds.append(lab)
                confs.append(sc)

        out = df.copy()
        out["prediction"] = preds
        out["confidence"] = np.round(confs, 4)
        st.dataframe(out.head(30), use_container_width=True)

        # download
        buf = io.StringIO()
        out.to_csv(buf, index=False)
        st.download_button("Download Predictions CSV", buf.getvalue(),
                           file_name="predictions.csv", mime="text/csv")

# --- TAB 3: Explain ---
with tab3:
    st.write("Sentence-level heatmap: **red = more fake**, **green = more real**.")
    text_exp = st.text_area("Text to explain", height=220, key="exp")
    if st.button("Explain", key="explain_btn"):
        if text_exp.strip():
            df_exp = sentence_explain(text_exp)
            if df_exp.empty:
                st.info("No sentences found.")
            else:
                # Render colored sentences
                html = " ".join([color_sentence_html(r.sentence, r.prob_fake) for _, r in df_exp.iterrows()])
                st.markdown(f'<div class="section">{html}</div>', unsafe_allow_html=True)
                st.markdown("##### Sentence Scores")
                st.dataframe(
                    df_exp[["sentence","label","prob_fake","conf"]]
                    .rename(columns={"prob_fake":"prob_fake(0=real,1=fake)"}),
                    use_container_width=True
                )

# --- TAB 4: Evaluate ---
with tab4:
    st.write("Upload a **labeled** CSV with columns: `text`, `label` (values: `real` or `fake`).")
    up_eval = st.file_uploader("Upload labeled CSV", type=["csv"], key="eval_csv")
    if up_eval is not None:
        eval_df = pd.read_csv(up_eval)
        # normalize
        text_col = "text" if "text" in eval_df.columns else eval_df.columns[0]
        label_col = "label" if "label" in eval_df.columns else None
        if label_col is None:
            st.error("No 'label' column found.")
        else:
            y_true, y_pred = [], []
            with st.spinner("Evaluating..."):
                for t, y in zip(eval_df[text_col].fillna(""), eval_df[label_col].astype(str).str.lower()):
                    if not t.strip():
                        y_true.append(y); y_pred.append("real"); continue
                    n_tokens = len(tok(t)["input_ids"])
                    if n_tokens <= 512:
                        lab, sc = predict_short(t)
                        if lab.lower()=="fake" and sc < threshold:
                            lab = "real"
                    else:
                        lab, sc, _, _ = predict_long(t, chunk_tokens=chunk_tokens)
                    y_true.append(y)
                    y_pred.append(lab.lower())

            # metrics
            labels = ["real","fake"]
            cm = pd.crosstab(pd.Series(y_true, name="Actual"),
                             pd.Series(y_pred, name="Predicted"),
                             rownames=["Actual"], colnames=["Predicted"]).reindex(index=labels, columns=labels, fill_value=0)
            acc = (np.array(y_true) == np.array(y_pred)).mean()
            # precision/recall/f1 (macro)
            pr, rc, f1s = [], [], []
            for cls in labels:
                tp = cm.loc[cls, cls]
                fp = cm[cls].sum() - tp
                fn = cm.loc[cls].sum() - tp
                precision = tp / (tp + fp) if (tp+fp)>0 else 0.0
                recall    = tp / (tp + fn) if (tp+fn)>0 else 0.0
                f1        = 2*precision*recall / (precision+recall) if (precision+recall)>0 else 0.0
                pr.append(precision); rc.append(recall); f1s.append(f1)
            macro_p = np.mean(pr); macro_r = np.mean(rc); macro_f1 = np.mean(f1s)

            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Accuracy", f"{acc:.2%}")
            with col2: st.metric("Macro Precision", f"{macro_p:.2%}")
            with col3: st.metric("Macro Recall", f"{macro_r:.2%}")
            with col4: st.metric("Macro F1", f"{macro_f1:.2%}")

            st.markdown("#### Confusion Matrix")
            st.dataframe(cm, use_container_width=True)
# ====== EXTRA FEATURES: URL SCRAPER + REPORT GENERATOR ======
from datetime import datetime
from fpdf import FPDF
import base64
from newspaper import Article

st.markdown("---")
st.header("🌐 URL Scraper & 🧾 Report Generator")

tab_scrape, tab_report = st.tabs(["🌐 URL Scraper", "🧾 Report Generator"])

# --- TAB 1: URL SCRAPER ---
with tab_scrape:
    st.write("Paste a **news article URL**. The app will extract text automatically and classify it.")
    url_input = st.text_input("Enter URL (e.g., https://www.bbc.com/news/...)", "")

    if st.button("Scrape & Classify", key="scrape_btn"):
        if not url_input.strip():
            st.warning("Please paste a valid URL.")
        else:
            with st.spinner("Fetching and analyzing..."):
                try:
                    article = Article(url_input)
                    article.download()
                    article.parse()
                    title = article.title
                    text = article.text.strip()

                    if not text:
                        st.error("Could not extract text. Some sites block scraping.")
                    else:
                        st.success(f"**Article Title:** {title}")
                        st.markdown("##### Article Preview")
                        st.write(text[:800] + ("..." if len(text) > 800 else ""))

                        # --- Model prediction ---
                        n_tokens = len(tok(text)["input_ids"])
                        if n_tokens <= 512:
                            label, score = predict_short(text)
                        else:
                            label, score, votes_df, raw_preds = predict_long(text)

                        # --- Display results ---
                        label_badge(label)
                        confidence_meter(score)
                        st.caption(f"Model: {MODEL_DIR.split('/')[-1]}  |  Length: {len(text.split())} words")

                        # --- Save context for report generator ---
                        st.session_state["last_pred"] = {
                            "source": "URL",
                            "title": title,
                            "url": url_input,
                            "text": text,
                            "label": label,
                            "score": score,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                except Exception as e:
                    st.error(f"Failed to fetch: {e}")   
 
# --- Gemini Integration ---
if "last_pred" in st.session_state:
    import google.generativeai as genai   

    with st.expander("🤖 Gemini Summary & Reasoning"):

        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            key="gemini_api_key"
        )

        if st.button("Ask Gemini"):

            try:

                pred = st.session_state.get("last_pred")

                if pred is None:
                    st.error("Please classify an article first.")
                    st.stop()

                genai.configure(api_key=api_key)
                

                prompt = f"""
                Summarize this article and explain why it was classified as {pred['label']}.

                Article:
                {pred['text'][:1500]}
                """

                model = genai.GenerativeModel(
                    "gemini-2.5-flash"
                )

                with st.spinner("Gemini analyzing..."):
                    response = model.generate_content(prompt)

                st.write(response.text)

            except Exception as e:
                st.error(str(e))                        

# --- TAB 2: REPORT GENERATOR ---
with tab_report:
    st.write("Generate a PDF/HTML report from your latest classification (single text or URL).")
    if "last_pred" not in st.session_state:
        st.info("No recent prediction found. Run a classification first.")
    else:
        data = st.session_state["last_pred"]
        st.markdown(f"**Last classified source:** {data['source']}  \n**Label:** {data['label']}  \n**Confidence:** {data['score']:.2%}")
        gen_pdf = st.button("Generate PDF Report", type="primary")
        gen_html = st.button("Generate HTML Report")

        def safe_pdf_text(text):
            return (
                str(text)
                .replace("“", '"')
                .replace("”", '"')
                .replace("‘", "'")
                .replace("’", "'")
                .replace("—", "-")
                .replace("–", "-")
                .replace("…", "...")
                .encode("latin-1", "replace")
                .decode("latin-1")
            )


        if gen_pdf:
            pdf = FPDF()
            pdf.add_page()

            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "Fake News Detection Report", new_x="LMARGIN", new_y="NEXT", align="C")
            pdf.ln(5)

            pdf.set_font("Helvetica", "", 12)

            pdf.cell(
                0,
                8,
                safe_pdf_text(f"Date: {data['timestamp']}"),
                new_x="LMARGIN",
                new_y="NEXT"
            )

            pdf.cell(
                0,
                8,
                safe_pdf_text(f"Source: {data['source']}"),
                new_x="LMARGIN",
                new_y="NEXT"
            )

            if "url" in data:
                pdf.multi_cell(
                    180,
                    8,
                    safe_pdf_text(f"URL: {data['url']}")
                )

            pdf.cell(
                0,
                8,
                safe_pdf_text(f"Model: {MODEL_DIR.split('/')[-1]}"),
                new_x="LMARGIN",
                new_y="NEXT"
            )

            pdf.cell(
                0,
                8,
                safe_pdf_text(f"Predicted Label: {data['label']}"),
                new_x="LMARGIN",
                new_y="NEXT"
            )

            pdf.cell(
                0,
                8,
                safe_pdf_text(f"Confidence: {data['score']:.2%}"),
                new_x="LMARGIN",
                new_y="NEXT"
            )

            pdf.ln(5)

            text_preview = data["text"][:1500].replace("\n", " ")

            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Text Preview", new_x="LMARGIN", new_y="NEXT")

            pdf.set_font("Helvetica", "", 11)

            pdf.multi_cell(
                180,
                8,
                safe_pdf_text(text_preview)
            )

            pdf.ln(5)

            pdf.set_font("Helvetica", "I", 10)

            pdf.multi_cell(
                180,
                8,
                safe_pdf_text(
                    "Note: This report was generated automatically using a fine-tuned transformer model for fake news classification."
                )
            )

            file_name = f"FakeNews_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

            pdf.output(file_name)

            st.success(f"Report saved as `{file_name}`")

            with open(file_name, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()

            href = (
                f'<a href="data:application/pdf;base64,{b64}" '
                f'download="{file_name}">📄 Download Report</a>'
            )

            st.markdown(href, unsafe_allow_html=True)

        if gen_html:
            # --- Create HTML ---
            html = f"""
            <html>
            <head><title>Fake News Report</title></head>
            <body style="font-family:Arial;padding:20px;">
            <h2>Fake News Detection Report</h2>
            <p><b>Date:</b> {data['timestamp']}</p>
            <p><b>Source:</b> {data['source']}</p>
            {'<p><b>URL:</b> ' + data['url'] + '</p>' if 'url' in data else ''}
            <p><b>Model:</b> {MODEL_DIR.split('/')[-1]}</p>
            <p><b>Predicted Label:</b> <span style="color:{'red' if data['label'].lower()=='fake' else 'green'}">{data['label']}</span></p>
            <p><b>Confidence:</b> {data['score']:.2%}</p>
            <h3>Text Preview:</h3>
            <p>{data['text'][:1500].replace('\n',' ') + ('...' if len(data['text'])>1500 else '')}</p>
            <hr><small>Generated automatically using a fine-tuned transformer model for fake news classification.</small>
            </body></html>
            """

            file_name = f"FakeNews_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(html)
            st.success(f"Report saved as `{file_name}`")
            with open(file_name, "r", encoding="utf-8") as f:
                b64 = base64.b64encode(f.read().encode()).decode()
                href = f'<a href="data:text/html;base64,{b64}" download="{file_name}">📄 Download HTML Report</a>'
                st.markdown(href, unsafe_allow_html=True)
# ====== LIVE NEWS FEED TAB ======
import feedparser, requests

st.markdown("---")
st.header("📰 Live News & Govt Feeds")

tab_news, tab_gov = st.tabs(["🌍 Latest News", "🏛️ Govt / PIB Updates"])

# --- 1) Public News headlines ---
with tab_news:
    st.write("Pulling latest headlines (Google News RSS). Click a headline to open it.")

    feeds = {
        "India": "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en",
        "World": "https://news.google.com/rss?hl=en&gl=US&ceid=US:en",
        "Technology": "https://news.google.com/rss/search?q=technology&hl=en-IN&gl=IN&ceid=IN:en",
    }

    choice = st.selectbox("Category", list(feeds.keys()))
    rss = feedparser.parse(feeds[choice])

    for entry in rss.entries[:10]:
        st.markdown(f"🔹 [{entry.title}]({entry.link})  \n<small>{entry.published}</small>", unsafe_allow_html=True)

# --- 2) PIB / Govt handles (robust RSS with fallbacks) ---
with tab_gov:
    st.write("Official PIB / MyGov recent posts (RSS with mirror rotation + news fallback)")

    import feedparser, requests
    from urllib.parse import quote

    # rotate through a few Nitter mirrors (these change—feel free to add/remove)
    NITTERS = [
        "https://nitter.net",
        "https://nitter.privacydev.net",
        "https://nitter.poast.org",
    ]

    HANDLES = {
        "PIB India (@PIB_India)": "PIB_India",
        "MyGov India (@mygovindia)": "mygovindia",
        "PIB FactCheck (@PIBFactCheck)": "PIBFactCheck",
    }

    who = st.selectbox("Choose handle", list(HANDLES.keys()), key="gov_handle")
    handle = HANDLES[who]

    @st.cache_data(ttl=300, show_spinner=False)
    def fetch_twitter_rss(handle: str):
        errors = []
        headers = {"User-Agent": "Mozilla/5.0"}
        for base in NITTERS:
            try:
                url = f"{base}/{handle}/rss"
                resp = requests.get(url, headers=headers, timeout=10)
                if resp.status_code == 200 and resp.content:
                    feed = feedparser.parse(resp.content)
                    if feed.entries:
                        return feed.entries, f"Source: {base}"
                errors.append(f"{base} -> {resp.status_code}")
            except Exception as e:
                errors.append(f"{base} -> {e}")
        return [], "; ".join(errors)

    @st.cache_data(ttl=300, show_spinner=False)
    def fetch_google_news(query: str):
        # Google News RSS fallback (English, India)
        url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(url)
        return feed.entries

    entries, debug = fetch_twitter_rss(handle)

    if entries:
        st.caption(debug)
        for e in entries[:8]:
            # Convert Nitter link to X.com for opening in browser
            x_link = e.link.replace("nitter.net", "x.com").replace("/statuses/", "/status/")
            title = getattr(e, "title", "(post)")
            published = getattr(e, "published", "")
            with st.container(border=True):
                st.markdown(f"**{title}**")
                if published:
                    st.caption(published)
                st.markdown(f"[Open on X]({x_link})")
                # Optional: classify post text with your model
                if st.button("Classify this post", key=f"classify_{hash(e.link)}"):
                    preview = title
                    n_tokens = len(tok(preview)["input_ids"])
                    if n_tokens <= 512:
                        lab, sc = predict_short(preview)
                    else:
                        lab, sc, _, _ = predict_long(preview)
                    label_badge(lab); confidence_meter(sc)
    else:
        st.warning("No items from Nitter mirrors (rate-limited or down). Showing latest official news instead.")
        # Fallback queries to official sources
        FALLBACKS = {
            "PIB India": "site:pib.gov.in",
            "MyGov India": "site:mygov.in",
            "PIB FactCheck": "PIB Fact Check India",
        }
        q = FALLBACKS["PIB India"] if "PIB India" in who else (
            FALLBACKS["MyGov India"] if "MyGov" in who else FALLBACKS["PIB FactCheck"]
        )
        news_items = fetch_google_news(q)
        if not news_items:
            st.error("Still no items. Try again in a minute or switch handles.")
        else:
            for e in news_items[:10]:
                title = getattr(e, "title", "")
                link = getattr(e, "link", "")
                pub = getattr(e, "published", "")
                with st.container(border=True):
                    st.markdown(f"**{title}**")
                    if pub:
                        st.caption(pub)
                    st.markdown(f"[Open]({link})")


# === DARK/LIGHT THEME TOGGLE ===
theme = st.sidebar.radio("🎨 Theme", ["Light", "Dark"], horizontal=True)
if theme == "Dark":
    st.markdown("""
    <style>
    body, .stApp { background-color:#111; color:#eee; }
    .badge.fake { background:#4c1d1d; color:#fff; }
    .badge.real { background:#1d4ed8; color:#fff; }
    .section { border-color:#444; }
    </style>
    """, unsafe_allow_html=True)


