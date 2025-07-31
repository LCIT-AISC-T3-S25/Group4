import streamlit as st
import requests

# URL of your running FastAPI backend
FASTAPI_URL = "http://localhost:8011/predict"

# Page config
st.set_page_config(page_title="RAG Model Q&A", layout="centered")
st.title("RAG Model Q&A")
st.write("Enter a question and get the model's response.")

# Input text box
question = st.text_area("Ask a question:")

if st.button("Get Answer"):
    if question.strip():
        payload = {"text": question}
        try:
            # Send request to FastAPI backend
            response = requests.post(FASTAPI_URL, json=payload)

            # Debug: show raw response
            st.write("Raw API response:", response.text)

            if response.status_code == 200:
                result = response.json()

                # Safely extract values
                input_text = result.get("input_text", question)
                answer = result.get("result", "No answer provided")

                # Display response
                st.subheader("Response")
                st.markdown(f"**Question:** {input_text}")

                # Handle types of answers
                if answer == "Relevant":
                    st.success(f"Answer: {answer}")
                elif answer == "Not relevant":
                    st.error(f"Answer: {answer}")
                else:
                    # Show the real chatbot answer
                    st.info(f"Answer: {answer}")

            else:
                st.error(f"Error {response.status_code}: {response.text}")

        except Exception as e:
            st.error(f"Could not connect to API: {e}")
    else:
        st.warning("Please enter a question before submitting.")
