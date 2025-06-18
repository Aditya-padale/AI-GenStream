import streamlit as st
import base64
import os
import magic
from PIL import Image
from dotenv import load_dotenv
import torch
import uuid

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# AnimateDiff imports
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_video  # Use export_to_video from AnimateDiff-Lightning
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini models
multimodal_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
image_gen_llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash-preview-image-generation")

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

# --- AnimateDiff text-to-video integration ---
# Load base models and motion loras from ByteDance repo
bases = {
    "ToonYou": "frankjoshua/toonyou_beta6",
    "epiCRealism": "emilianJR/epiCRealism"
}

step_loaded = None
base_loaded = "ToonYou"
motion_loaded = None

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# Load initial pipeline and scheduler once
pipe = AnimateDiffPipeline.from_pretrained(bases[base_loaded], torch_dtype=dtype).to(device)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")

from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor

safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker").to(device)
feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")

def check_nsfw_images(images: list[Image.Image]) -> list[bool]:
    safety_checker_input = feature_extractor(images, return_tensors="pt").to(device)
    has_nsfw_concepts = safety_checker(images=images, clip_input=safety_checker_input.pixel_values.to(device))
    return has_nsfw_concepts

def generate_video(prompt, base, motion, step):
    global step_loaded, base_loaded, motion_loaded

    # Load weights for steps
    if step_loaded != step:
        repo = "ByteDance/AnimateDiff-Lightning"
        ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
        pipe.unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device), strict=False)
        step_loaded = step

    # Load base model
    if base_loaded != base:
        pipe.unet.load_state_dict(torch.load(hf_hub_download(bases[base], "unet/diffusion_pytorch_model.bin"), map_location=device), strict=False)
        base_loaded = base

    # Load motion lora weights
    if motion_loaded != motion:
        pipe.unload_lora_weights()
        if motion != "":
            pipe.load_lora_weights(motion, adapter_name="motion")
            pipe.set_adapters(["motion"], [0.7])
        motion_loaded = motion

    output = pipe(prompt=prompt, guidance_scale=1.0, num_inference_steps=step)

    # NSFW check
    if check_nsfw_images([output.frames[0][0]])[0]:
        st.warning("⚠️ NSFW content detected. Aborting generation.")
        return None

    name = str(uuid.uuid4()).replace("-", "")
    video_path = f"/tmp/{name}.mp4"
    export_to_video(output.frames[0], video_path, fps=10)
    return video_path

# --- Streamlit UI ---
st.set_page_config(page_title="Multimodal AI with Gemini & AnimateDiff", layout="centered")
st.title("🌐 Multimodal AI with Gemini & AnimateDiff (ByteDance)")

option = st.radio("Choose a task", [
    "🖼️ Image Q&A and Transformation",
    "🎤 Audio Transcription",
    "🎥 Video Analysis",
    "🎨 AI Image Generation",
    "🎬 Text-to-Video (AnimateDiff)"
])

if option == "🖼️ Image Q&A and Transformation":
    st.write("📸 Upload an image to analyze, transform, or ask questions about it!")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    # ... (keep your existing code unchanged here) ...

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        tab1, tab2 = st.tabs(["📝 Ask & Analyze", "🎨 Transform"])

        with tab1:
            operation = st.radio("Choose operation", ["Ask a question", "Analyze image"], horizontal=True)
            if operation == "Ask a question":
                question = st.text_area("What would you like to know about the image?", "What can you tell me about this image?")
                prompt = question
            else:
                analysis_type = st.selectbox("What to analyze?", ["Detailed description", "Objects and elements", "Colors and composition", "Emotions and mood", "Text content"])
                prompt = f"Analyze this image focusing on {analysis_type.lower()}. Provide a detailed response."

            if st.button("Analyze", key="analyze_btn"):
                try:
                    uploaded_file.seek(0)
                    encoded_img = encode_base64(uploaded_file.read())

                    message = HumanMessage(content=[
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{encoded_img}"}
                    ])
                    response = multimodal_llm.invoke([message])
                    with st.expander("Analysis Results", expanded=True):
                        st.markdown(response.content)
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")

        with tab2:
            style = st.selectbox("Choose transformation style", ["Anime/Ghibli", "Oil painting", "Watercolor", "Digital art", "Pencil sketch", "Pixel art", "Photorealistic", "Custom"])
            if style == "Custom":
                transform_prompt = st.text_area("Describe the style you want", "Transform this image into...")
            else:
                transform_prompt = f"Transform this image into {style} style. Maintain core composition but apply new style."

            if st.button("Transform", key="transform_btn"):
                try:
                    uploaded_file.seek(0)
                    encoded_img = encode_base64(uploaded_file.read())

                    response = image_gen_llm.invoke(
                        [HumanMessage(content=[
                            {"type": "text", "text": transform_prompt},
                            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{encoded_img}"}
                        ])],
                        generation_config=dict(response_modalities=["TEXT", "IMAGE"])
                    )
                    img_base64 = extract_image_base64(response)
                    if img_base64:
                        img_bytes = base64.b64decode(img_base64)
                        st.success("✨ Transformation complete!")
                        st.image(img_bytes, caption="Transformed Image", use_column_width=True)
                        st.download_button("⬇️ Download Transformed Image", img_bytes, file_name=f"transformed_{style.lower().replace('/', '_')}.png", mime="image/png")
                    else:
                        st.error("Failed to generate transformed image")
                except Exception as e:
                    st.error(f"Error during transformation: {str(e)}")
                    st.error("Try a different style or image")
    else:
        st.info("👆 Upload an image to get started!")

elif option == "🎤 Audio Transcription":
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])
    prompt = st.text_input("Enter prompt for audio analysis", "Please transcribe this audio and provide context.")
    if uploaded_file and st.button("Transcribe Audio"):
        try:
            mime_type = get_mime(uploaded_file)
            uploaded_file.seek(0)
            encoded_audio = encode_base64(uploaded_file.read())
            message = HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "media", "data": encoded_audio, "mime_type": mime_type}
            ])
            response = multimodal_llm.invoke([message])
            st.success(response.content)
        except Exception as e:
            st.error(f"Error in transcription: {str(e)}")

elif option == "🎥 Video Analysis":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov"])
    prompt = st.text_input("Enter prompt for video analysis", "Analyze this video and describe what's happening.")
    if uploaded_file and st.button("Analyze Video"):
        try:
            mime_type = get_mime(uploaded_file)
            uploaded_file.seek(0)
            encoded_video = encode_base64(uploaded_file.read())
            message = HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "media", "data": encoded_video, "mime_type": mime_type}
            ])
            response = multimodal_llm.invoke([message])
            st.success(response.content)
        except Exception as e:
            st.error(f"Error in video analysis: {str(e)}")

elif option == "🎨 AI Image Generation":
    prompt = st.text_input("Enter an image prompt")
    if st.button("Generate Image"):
        try:
            message = HumanMessage(content=prompt)
            response = image_gen_llm.invoke(
                [message],
                generation_config=dict(response_modalities=["TEXT", "IMAGE"])
            )
            img_base64 = extract_image_base64(response)
            if img_base64:
                img_bytes = base64.b64decode(img_base64)
                st.image(img_bytes, caption="Generated Image", use_column_width=True)
            else:
                st.error("❌ Failed to generate image.")
        except Exception as e:
            st.error(f"Error generating image: {str(e)}")

elif option == "🎬 Text-to-Video (AnimateDiff)":
    st.write("Generate short animated videos from text prompts using ByteDance AnimateDiff-Lightning.")
    prompt = st.text_area("Enter video prompt", "A cat playing piano in a colorful room")
    select_base = st.selectbox(
        "Select base model",
        options=["ToonYou", "epiCRealism"],
        index=0
    )
    select_motion = st.selectbox(
        "Select motion style",
        options=[
            ("Default", ""),
            ("Zoom in", "guoyww/animatediff-motion-lora-zoom-in"),
            ("Zoom out", "guoyww/animatediff-motion-lora-zoom-out"),
            ("Tilt up", "guoyww/animatediff-motion-lora-tilt-up"),
            ("Tilt down", "guoyww/animatediff-motion-lora-tilt-down"),
            ("Pan left", "guoyww/animatediff-motion-lora-pan-left"),
            ("Pan right", "guoyww/animatediff-motion-lora-pan-right"),
            ("Roll left", "guoyww/animatediff-motion-lora-rolling-anticlockwise"),
            ("Roll right", "guoyww/animatediff-motion-lora-rolling-clockwise"),
        ],
        index=0,
        format_func=lambda x: x[0]
    )
    step = st.select_slider("Select inference steps", options=[1, 2, 4, 8], value=4)

    if st.button("Generate Animation"):
        try:
            with st.spinner("Generating animation... This may take some time."):
                video_path = generate_video(prompt, select_base, select_motion[1], step)
                if video_path:
                    st.video(video_path)
                    with open(video_path, "rb") as f:
                        st.download_button("⬇️ Download Video", data=f, file_name="animated_video.mp4", mime="video/mp4")
        except Exception as e:
            st.error(f"Error generating video: {str(e)}")
