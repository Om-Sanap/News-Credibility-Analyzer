import streamlit as st
import joblib

vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

st.title("Fake News Detector")
st.write("Enter a News Article below to check whether it is Fake or Real. ")

news_input = st.text_area("News Article", height=200)
inputn = news_input

if st.button("Check News"):
    if inputn.strip():
        transform_input = vectorizer.transform([inputn])
        prediction = model.predict(transform_input)

        if prediction[0] == 1:
            st.success("The News is Real! ")
        else:
            st.error("The News is Fake! ")
    else:
        st.warning("Please enter some text to Analyze. ") 
