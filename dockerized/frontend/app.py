import streamlit as st
import requests
import re
import string

import os

API_URL = os.getenv("API_URL", "http://localhost:8000")
  # Your FastAPI backend URL

st.title("Sentiment Analysis with LIME Explanation")

text = st.text_input("Enter text for sentiment analysis:")

def normalize_token(tok):
    return tok.lower().strip(string.punctuation)

# Inverted sentiment logic: flip the meaning
def color_for_score_inverted(score):
    if score > 0:
        return "#f3a6a6"  # originally positive → now treat as negative
    elif score < 0:
        return "#a6f3a6"  # originally negative → now treat as positive
    else:
        return None  # no highlight

if text:
    # Get prediction
    pred_res = requests.post(f"{API_URL}/predict", json={"text": text})
    if pred_res.status_code == 200:
        pred_data = pred_res.json()
        label = pred_data.get("label", "Unknown")
        confidence = pred_data.get("confidence", 0)
        st.markdown(f"**Predicted Sentiment:** {label} (Confidence: {confidence:.2f})")
    else:
        st.error("Prediction API error")

    # Get LIME explanation
    explain_res = requests.post(f"{API_URL}/explain", json={"text": text})
    if explain_res.status_code == 200:
        explain_data = explain_res.json()
        explanation = explain_data.get("explanation", {})

        if explanation and isinstance(explanation, dict):
            words = re.findall(r'\w+|\W+', text)
            highlighted_words = []
            for w in words:
                w_norm = normalize_token(w)
                score = explanation.get(w_norm, 0)
                color = color_for_score_inverted(score)
                if color:
                    highlighted_words.append(
                        f'<span style="background-color:{color}; padding:2px 4px; border-radius:3px;">{w}</span>'
                    )
                else:
                    highlighted_words.append(w)

            highlighted_text = "".join(highlighted_words)
            st.markdown("### LIME Explanation (highlighted text)")
            st.markdown(highlighted_text, unsafe_allow_html=True)

            # Inverted Legend
            legend_html = """
            <div style="margin-top:10px; font-size:14px;">
                <span style="background-color:#f3a6a6; padding:5px; border-radius:3px; margin-right:10px;">Positive</span>
                <span style="background-color:#a6f3a6; padding:5px; border-radius:3px; margin-right:10px;">Negative</span>
                <span style="padding:5px;">Neutral / No Highlight</span>
            </div>
            """
            st.markdown(legend_html, unsafe_allow_html=True)
        else:
            st.info("No explanation available or unexpected format.")
    else:
        st.error("Explanation API error")
