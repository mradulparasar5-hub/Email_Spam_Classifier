"""Streamlit web app for the spam classifier.

Run with:  streamlit run app.py
"""
import streamlit as st

from src.predict import load_model, predict_message

st.set_page_config(page_title="Email / SMS Spam Classifier", page_icon="📧")


@st.cache_resource
def get_model():
    return load_model()


st.title("📧 Email / SMS Spam Classifier")
st.write("Paste a message below and the model will tell you whether it looks like **spam** or **ham** (legitimate).")

examples = {
    "— choose an example —": "",
    "Spam example": "WINNER!! You have been selected to receive a £900 prize reward. Call 09061701461 now to claim.",
    "Ham example": "Hey, are we still meeting for lunch tomorrow? Let me know what time works for you.",
}
choice = st.selectbox("Try an example", list(examples))
text = st.text_area("Message", value=examples[choice], height=160)

if st.button("Classify", type="primary"):
    if not text.strip():
        st.warning("Please enter a message first.")
    else:
        result = predict_message(get_model(), text)
        prob = result["spam_probability"]
        if result["label"] == "spam":
            st.error(f"🚨 SPAM  —  {prob:.1%} confidence")
        else:
            st.success(f"✅ Not spam (ham)  —  {1 - prob:.1%} confidence")
        st.progress(prob, text=f"Spam probability: {prob:.1%}")
