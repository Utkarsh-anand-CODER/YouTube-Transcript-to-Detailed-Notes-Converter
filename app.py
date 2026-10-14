
import os
import time

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import errors

# 1. Load environment variables
load_dotenv()

# 2. Configure Streamlit page
st.set_page_config(page_title="Gemini Q&A Demo")
st.header("Gemini LLM Application")

# 3. Read API key
api_key = os.getenv("Gemini_API_Key")

if not api_key:
    st.error("API key not found. Please check your .env file.")
    st.stop()

# 4. Create Gemini client once and reuse it
@st.cache_resource
def get_gemini_client(key):
    return genai.Client(api_key=key)

client = get_gemini_client(api_key)

# 5. Select a model available to your API project
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


# 6. Generate a streaming response with retry handling
def stream_gemini_response(question):
    for attempt in range(3):
        try:
            stream = client.models.generate_content_stream(
                model=MODEL_NAME,
                contents=question,
            )

            # Display each chunk as soon as it arrives
            for chunk in stream:
                if chunk.text:
                    yield chunk.text

            return

        except errors.APIError as e:
            # Retry temporary server and rate-limit errors
            if e.code in (429, 500, 502, 503, 504):
                if attempt < 2:
                    wait_time = 2 ** (attempt + 1)
                    time.sleep(wait_time)
                    continue

                yield (
                    "\n\nGemini is temporarily busy. "
                    "Please try again in a few minutes."
                )
                return

            yield f"\n\nGemini API error: {e}"
            return

        except Exception as e:
            yield f"\n\nUnexpected error: {e}"
            return


# 7. Streamlit user interface
user_input = st.text_input("Enter your question:")

if st.button("Ask the question"):
    if user_input.strip():
        st.subheader("Gemini's response")

        with st.spinner("Connecting to Gemini..."):
            st.write_stream(
                stream_gemini_response(user_input.strip())
            )
    else:
        st.warning("Please enter a question first.")
