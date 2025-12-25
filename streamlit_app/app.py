import streamlit as st
import requests
import threading
from api.main import app as fastapi_app
import uvicorn

def run_fastapi():
    uvicorn.run(
        fastapi_app,
        host="0.0.0.0",
        port=8000,
        log_level="warning"
    )
# Start FastAPI server in a separate thread
threading.Thread(target=run_fastapi, daemon=True).start()


# FastAPI endpoint
API_URL = "http://127.0.0.1:8000/predict"

# Page configuration
st.set_page_config(
    page_title="English NER",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 English Named Entity Recognition")
st.write("Extract named entities from English text using a deep learning model.")

# Text input
text_input = st.text_area(
    "English Text",
    placeholder="Type your text here...",
    height=150
)

# Extract button
if st.button("Extract Entities"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        try:
            # Spinner while calling the API
            with st.spinner("Extracting entities..."):
                response = requests.post(
                    API_URL,
                    json={"text": text_input},
                    timeout=15
                )

            if response.status_code == 200:
                result = response.json()
                entities = result.get("entities", [])

                if entities:
                    st.subheader("Named Entities Found:")
                    for ent in entities:
                        st.markdown(f"- **Token:** `{ent['token']}` → **Entity:** `{ent['entity']}`")
                else:
                    st.info("No named entities detected in the text.")

            else:
                st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")

        except requests.exceptions.RequestException:
            st.error("❌ Could not connect to the FastAPI server.")
