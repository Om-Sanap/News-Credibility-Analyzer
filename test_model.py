import streamlit as st
from transformers import pipeline

st.write("Loading model...")

clf = pipeline(
    "text-classification",
    model="hf_fake_news/best"
)

st.success("Model loaded!")