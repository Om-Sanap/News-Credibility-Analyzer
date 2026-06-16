# News Credibility Analyzer

A fake news detection system built using a fine-tuned RoBERTa model.

An NLP-based web application that analyzes news articles and predicts whether content resembles patterns found in real or fake news datasets.

Built using Hugging Face Transformers, Streamlit, and Google's Gemini API.

## Features

* Fake News Classification using a fine-tuned Transformer
* News Article URL Scraping
* Confidence Score Visualization
* AI-Powered Summaries and Reasoning (Gemini)
* PDF Report Generation
* HTML Report Generation
* Live News Feed Integration
* Interactive Streamlit Dashboard

## Tech Stack

* Python
* Streamlit
* Hugging Face Transformers
* PyTorch
* Google Gemini API
* newspaper3k
* FPDF

## Project Architecture

User Input / URL
↓
Article Extraction
↓
Transformer Classification
↓
Confidence Scoring
↓
Gemini Explanation
↓
Report Generation

## Installation

```bash
git clone https://github.com/Techgeek02/FakeNewsDetection.git
cd FakeNewsDetection

pip install -r requirements.txt
streamlit run app_hf_fancy.py
```

## Screenshots

![alt text](pdf_report.png) ![alt text](gemini_overview.png) ![alt text](gov_feed.png) ![alt text](url_scrapper.png) ![alt text](home.png)

## Limitations

This system identifies patterns associated with misinformation based on its training dataset. It does not independently verify factual claims against trusted external sources.

## Future Improvements

* Fact-check API integration
* Retrieval-Augmented Generation (RAG)
* Multilingual Support
* Explainable AI (SHAP/LIME)
* Social Media Analysis

# News Credibility Analyzer

🚀 Live Demo: https://news-credibility-analyzer101.streamlit.app

🤗 Hugging Face Model:
https://huggingface.co/bunpine/news-credibility-analyzer

📂 GitHub Repository:
https://github.com/Om-Sanap/News-Credibility-Analyzer
