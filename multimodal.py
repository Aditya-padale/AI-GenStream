import streamlit as st
import base64
import os
import magic
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import PyPDF2
from docx import Document
import requests
from io import BytesIO
from PIL import Image
import tempfile
import shutil
import subprocess
import yt_dlp

# --- Configuration ---
st.set_page_config(page_title="✨ Multimodal AI Playground", layout="centered")
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# --- Models ---
multimodal_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
image_gen_llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash-preview-image-generation")

# --- Helper Functions ---
def get_mime(file):
    return magic.from_buffer(file.read(2048), mime=True)

def encode_base64(file):
    return base64.b64encode(file).decode("utf-8")

def extract_image_base64(response):
    for block in response.content:
        if isinstance(block, dict) and "image_url" in block:
            url = block["image_url"]["url"]
            if "," in url:
                return url.split(",")[1]
    return None

# --- CSS Styling ---
st.markdown('''<style>body {background: linear-gradient(135deg, #f8fafc 0%, #e0e7ff 100%) !important;} .main {background: rgba(255,255,255,0.95) !important; border-radius: 18px; box-shadow: 0 4px 32px 0 rgba(60,72,100,0.08); padding: 2rem;} footer {visibility: hidden;} .stButton>button, .stDownloadButton>button {border-radius: 8px; font-weight: 600;} </style>''', unsafe_allow_html=True)

# --- Session History ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- Header ---
st.markdown("""
<div style='text-align:center;'>
    <h1 style='color:#6366f1;'>✨ Multimodal AI Playground</h1>
</div>
""", unsafe_allow_html=True)

# --- Feature Selection ---
option = st.radio("Choose a task", [
    "🖼️ Image Q&A and Transformation",
    "🎤 Audio Transcription",
    "🎥 Video Analysis",
    "🎨 AI Image Generation",
    "📄 Document Analysis"
])

# --- Image Q&A and Transformation ---
if option == "🖼️ Image Q&A and Transformation":
    st.subheader("Upload and analyze an image")
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    image_url = st.text_input("...or paste an image URL")

    encoded_img = None
    image_error = None
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        uploaded_file.seek(0)
        encoded_img = encode_base64(uploaded_file.read())
    elif image_url:
        try:
            resp = requests.get(image_url, timeout=10)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content))
            img_format = img.format if img.format else 'JPEG'
            buffered = BytesIO()
            img.save(buffered, format=img_format)
            encoded_img = base64.b64encode(buffered.getvalue()).decode("utf-8")
            st.image(img, caption="Image from URL", use_column_width=True)
        except Exception as e:
            image_error = f"Could not load image from URL: {e}"
            st.error(image_error)

    if encoded_img and not image_error:
        tab1, tab2 = st.tabs(["📝 Ask & Analyze", "🎨 Transform"])
        with tab1:
            choice = st.radio("Choose", ["Ask a question", "Image Analysis"], horizontal=True)
            if choice == "Ask a question":
                prompt = st.text_area("Your question", "What is in the image?")
            else:
                analysis = st.selectbox("Analyze", ["Objects", "Emotions", "Text", "Color"], index=0)
                prompt = f"Describe the {analysis.lower()} in this image."
            if st.button("Analyze Image"):
                message = HumanMessage(content=[{"type": "text", "text": prompt},
                                              {"type": "image_url", "image_url": f"data:image/jpeg;base64,{encoded_img}"}])
                response = multimodal_llm.invoke([message])
                st.session_state.history.append({"input": prompt, "response": response.content})
                st.success(response.content)
        with tab2:
            style = st.selectbox("Choose style", ["Anime", "Oil painting", "Sketch", "Pixel art", "Photorealistic", "Custom"])
            if style == "Custom":
                transform_prompt = st.text_area("Describe style", "Transform into...")
            else:
                transform_prompt = f"Transform this image into {style} style."
            if st.button("Transform Image"):
                response = image_gen_llm.invoke([
                    HumanMessage(content=[
                        {"type": "text", "text": transform_prompt},
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{encoded_img}"}
                    ])], generation_config=dict(response_modalities=["TEXT", "IMAGE"]))
                img_base64 = extract_image_base64(response)
                if img_base64:
                    img_bytes = base64.b64decode(img_base64)
                    st.image(img_bytes, caption="Transformed Image")
                    st.download_button("Download Image", img_bytes, file_name="transformed.png", mime="image/png")
                else:
                    st.error("Failed to generate image.")

# --- Audio Transcription ---
elif option == "🎤 Audio Transcription":
    st.subheader("Upload audio for transcription")
    uploaded_file = st.file_uploader("Upload Audio", type=["mp3", "wav"])
    prompt = st.text_input("Prompt for transcription", "Please transcribe this.")
    if uploaded_file and st.button("Transcribe"):
        mime = get_mime(uploaded_file)
        uploaded_file.seek(0)
        encoded = encode_base64(uploaded_file.read())
        message = HumanMessage(content=[{"type": "text", "text": prompt},
                                        {"type": "media", "data": encoded, "mime_type": mime}])
        response = multimodal_llm.invoke([message])
        st.session_state.history.append({"input": prompt, "response": response.content})
        st.success(response.content)

# --- Video Analysis ---
elif option == "🎥 Video Analysis":
    st.subheader("Upload video for analysis")
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov"])
    video_url = st.text_input("...or paste a video URL (YouTube, Instagram, or direct .mp4/.mov)")
    prompt = st.text_input("Prompt for video", "Describe what is happening in the video.")
    video_path = None
    video_error = None
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        st.video(video_path)
    elif video_url:
        try:
            if any(x in video_url for x in ["youtube.com", "youtu.be", "instagram.com", "facebook.com", "tiktok.com"]):
                # Use yt_dlp Python library to download video
                with tempfile.TemporaryDirectory() as tmpdir:
                    ydl_opts = {
                        'format': 'mp4',
                        'outtmpl': f'{tmpdir}/video.%(ext)s',
                        'quiet': True,
                        'noplaylist': True,
                    }
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(video_url, download=True)
                        ext = info.get('ext', 'mp4')
                        candidate = os.path.join(tmpdir, f'video.{ext}')
                        if os.path.exists(candidate):
                            # Copy to a persistent temp file
                            with open(candidate, 'rb') as src, tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as dst:
                                shutil.copyfileobj(src, dst)
                                video_path = dst.name
                            st.video(video_path)
                        else:
                            video_error = "Could not download video from the provided link."
            elif video_url.lower().endswith((".mp4", ".mov", ".webm")):
                resp = requests.get(video_url, timeout=15, stream=True)
                resp.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    shutil.copyfileobj(resp.raw, tmp_file)
                    video_path = tmp_file.name
                st.video(video_path)
            else:
                video_error = "Unsupported video URL. Please provide a direct video file link or a YouTube/Instagram link."
        except Exception as e:
            video_error = f"Could not load video from URL: {e}"
            st.error(video_error)
    if video_path and not video_error:
        if st.button("Analyze Video"):
            try:
                with open(video_path, "rb") as f:
                    video_bytes = f.read()
                mime = get_mime(BytesIO(video_bytes))
                encoded = encode_base64(video_bytes)
                message = HumanMessage(content=[{"type": "text", "text": prompt},
                                                {"type": "media", "data": encoded, "mime_type": mime}])
                response = multimodal_llm.invoke([message])
                st.session_state.history.append({"input": prompt, "response": response.content})
                st.success(response.content)
            except Exception as e:
                st.error(f"Error analyzing video: {e}")

# --- AI Image Generation ---
elif option == "🎨 AI Image Generation":
    st.subheader("Generate an image from text prompt")
    prompt = st.text_input("Describe the image")
    if st.button("Generate"):
        message = HumanMessage(content=prompt)
        response = image_gen_llm.invoke([message], generation_config=dict(response_modalities=["TEXT", "IMAGE"]))
        img_base64 = extract_image_base64(response)
        if img_base64:
            img_bytes = base64.b64decode(img_base64)
            st.image(img_bytes, caption="Generated Image")
        else:
            st.error("Failed to generate image.")

# --- Document Analysis ---
elif option == "📄 Document Analysis":
    st.subheader("Upload a document for analysis")
    uploaded_file = st.file_uploader("Upload PDF or Word", type=["pdf", "docx"])
    prompt = st.text_input("What do you want to extract/analyze?", "Summarize the document.")
    doc_text = ""
    if uploaded_file:
        if uploaded_file.name.lower().endswith(".pdf"):
            try:
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                doc_text = "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
            except Exception as e:
                st.error(f"Failed to read PDF: {e}")
        elif uploaded_file.name.lower().endswith(".docx"):
            try:
                doc = Document(uploaded_file)
                doc_text = "\n".join([para.text for para in doc.paragraphs])
            except Exception as e:
                st.error(f"Failed to read DOCX: {e}")
    if doc_text:
        if st.button("Analyze Document"):
            user_prompt = f"{prompt}\n\nDocument Content:\n{doc_text[:4000]}"  # Limit to 4000 chars for LLM
            message = HumanMessage(content=user_prompt)
            response = multimodal_llm.invoke([message])
            st.session_state.history.append({"input": prompt, "response": response.content})
            st.success(response.content)

# --- History View ---
if st.checkbox("Show Session History"):
    for i, h in enumerate(reversed(st.session_state.history)):
        st.markdown(f"**{i+1}. Prompt:** {h['input']}")
        st.markdown(f"**Response:** {h['response']}")
        st.markdown("---")

# --- Footer ---
st.markdown("""
<hr style='margin-top:2em;'>
<div style='text-align:center;'>
    Crafted with ❤️ by <b style='color:#0ea5e9;'>Aditya Padale</b>
</div>
""", unsafe_allow_html=True)
