"""
app.py
-------
Simple Streamlit app: paste a news article, click Predict, see the result.

Run with:
    streamlit run app.py
"""

import os
import streamlit as st

from src.predict import predict_news
from src.utils import MODEL_PATH

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detector")
st.write("Paste a news headline or article below and check if it looks Real or Fake.")

MODEL_READY = os.path.exists(MODEL_PATH)

if not MODEL_READY:
    st.error("No trained model found. Run `python train_model.py` first, then reload this page.")
    st.stop()

text_input = st.text_area("News article text", height=200, placeholder="Paste the article here...")

if st.button("Predict", type="primary"):
    if not text_input or not text_input.strip():
        st.warning("Please paste some text first.")
    else:
        with st.spinner("Analyzing..."):
            try:
                result = predict_news(text_input)
            except ValueError as e:
                st.error(str(e))
                st.stop()

        if result["label"] == "Real":
            st.success(f"✅ Prediction: **REAL NEWS**")
        else:
            st.error(f"🚨 Prediction: **FAKE NEWS**")

        if result["confidence"] is not None:
            st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
            st.progress(result["confidence"])

        if result["prob_real"] is not None:
            col1, col2 = st.columns(2)
            col1.metric("P(Real)", f"{result['prob_real'] * 100:.1f}%")
            col2.metric("P(Fake)", f"{result['prob_fake'] * 100:.1f}%")
