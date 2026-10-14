
import os
import time
from urllib.parse import urlparse, parse_qs

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import errors
from youtube_transcript_api import YouTubeTranscriptApi

# 1. Load environment variables
load_dotenv()

# 2. Configure Streamlit
st.set_page_config(
    page_title="YouTube Video Summarizer",
    page_icon="📺",
    layout="centered"
)

st.title("YouTube Transcript to Detailed Notes Converter")
st.write("Generate concise, organized notes from YouTube videos.")

# 3. Configure Gemini API
api_key = os.getenv("Gemini_API_Key") or os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("API key not found. Please check your .env file.")
    st.stop()

client = genai.Client(api_key=api_key)

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# 4. Summarization prompt
PROMPT = """
You are an expert YouTube video summarizer.

Summarize the transcript into clear, useful study notes.

Requirements:
- Keep the summary within 250 words.
- Use headings and bullet points.
- Highlight important concepts, facts, and takeaways.
- Avoid repetition.
- Do not invent information absent from the transcript.

TRANSCRIPT:
"""


# 5. Extract video ID from supported YouTube URLs
def extract_video_id(video_url):
    parsed = urlparse(video_url.strip())
    host = (parsed.hostname or "").lower()

    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]

    elif host == "youtube.com" or host.endswith(".youtube.com"):
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        else:
            parts = parsed.path.strip("/").split("/")
            if len(parts) >= 2 and parts[0] in (
                "embed", "shorts", "live"
            ):
                video_id = parts[1]
            else:
                return None
    else:
        return None

    if (
        len(video_id) == 11
        and all(c.isalnum() or c in "_-" for c in video_id)
    ):
        return video_id

    return None


# 6. Extract YouTube transcript
def extract_transcript_details(video_id):
    transcript_api = YouTubeTranscriptApi()

    transcript = transcript_api.fetch(
        video_id,
        languages=["en", "en-US", "en-GB"]
    )

    return " ".join(
        snippet.text for snippet in transcript
    )


# 7. Generate notes with retries and streaming
def stream_gemini_content(transcript_text):
    prompt = PROMPT + transcript_text

    for attempt in range(3):
        try:
            stream = client.models.generate_content_stream(
                model=MODEL_NAME,
                contents=prompt
            )

            for chunk in stream:
                if chunk.text:
                    yield chunk.text

            return

        except errors.APIError as e:
            if e.code in (429, 500, 502, 503, 504):
                if attempt < 2:
                    wait_seconds = 2 ** (attempt + 1)
                    time.sleep(wait_seconds)
                    continue

                yield (
                    "\n\n**Gemini is temporarily unavailable.** "
                    "Please try again in a few minutes."
                )
                return

            yield f"\n\n**Gemini API error:** {e}"
            return

        except Exception as e:
            yield f"\n\n**Unexpected error:** {e}"
            return


# 8. Streamlit interface
youtube_url = st.text_input(
    "Enter YouTube video link:",
    placeholder="https://www.youtube.com/watch?v=..."
)

video_id = (
    extract_video_id(youtube_url)
    if youtube_url.strip()
    else None
)

if youtube_url.strip():
    if video_id:
        thumbnail_url = (
            f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
        )
        st.image(
            thumbnail_url,
            caption="YouTube video thumbnail",
            use_container_width=True
        )
    else:
        st.warning("Please enter a valid YouTube video URL.")


# 9. Fetch transcript and generate notes
if st.button("Generate Detailed Notes", type="primary"):
    if not youtube_url.strip():
        st.warning("Please enter a YouTube video URL.")

    elif not video_id:
        st.error("The YouTube URL is invalid or unsupported.")

    else:
        try:
            with st.spinner("Fetching video transcript..."):
                transcript_text = extract_transcript_details(video_id)

            if not transcript_text.strip():
                st.error("No transcript was found for this video.")
                st.stop()

            st.success("Transcript fetched successfully.")

            st.subheader("Detailed Notes")

            # Stream generated text into the interface
            notes = st.write_stream(
                stream_gemini_content(transcript_text)
            )

            if notes and "Gemini is temporarily unavailable." not in notes:
                st.download_button(
                    label="Download Notes",
                    data=notes,
                    file_name=f"{video_id}_notes.txt",
                    mime="text/plain"
                )

        except Exception as e:
            st.error("Could not process the YouTube video.")

            error_message = str(e)

            if "transcript" in error_message.lower():
                st.info(
                    "The video may not have accessible captions. "
                    "Try a video with subtitles enabled."
                )
            else:
                st.caption(f"Technical details: {error_message}")
