import streamlit as st
import speech_recognition as sr
import google.generativeai as genai
import requests
import io
import base64
from audio_recorder_streamlit import audio_recorder
import tempfile
import os
from pathlib import Path
import json
import time
from typing import List, Dict

# Configure page
st.set_page_config(
    page_title="Advanced AI Voice Agent",
    page_icon="🎤",
    layout="wide"
)

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'audio_response' not in st.session_state:
    st.session_state.audio_response = None
if 'cloned_voices' not in st.session_state:
    st.session_state.cloned_voices = []
if 'reader_audio' not in st.session_state:
    st.session_state.reader_audio = None

# Default API keys (you can modify these)
DEFAULT_ELEVENLABS_KEY = ""
DEFAULT_GEMINI_KEY = ""

def setup_apis():
    """Setup API keys and configurations"""
    st.sidebar.title("🔧 Configuration")

    # Pre-filled API keys
    gemini_api_key = st.sidebar.text_input(
        "Gemini API Key",
        value=DEFAULT_GEMINI_KEY,
        type="password",
        help="Get your API key from https://makersuite.google.com/app/apikey"
    )

    elevenlabs_api_key = st.sidebar.text_input(
        "ElevenLabs API Key",
        value=DEFAULT_ELEVENLABS_KEY,
        type="password",
        help="Get your API key from https://elevenlabs.io"
    )

    return gemini_api_key, elevenlabs_api_key

def get_available_voices(api_key):
    """Get all available voices from ElevenLabs"""
    try:
        url = "https://api.elevenlabs.io/v1/voices"
        headers = {"xi-api-key": api_key}

        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            voices_data = response.json()
            voices = {}
            for voice in voices_data.get('voices', []):
                voices[voice['name']] = voice['voice_id']
            return voices
        return {}
    except Exception as e:
        st.error(f"Error fetching voices: {str(e)}")
        return {}

def initialize_gemini(api_key):
    """Initialize Gemini AI with 2.0 Flash model"""
    if api_key:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        return model
    return None

def speech_to_text(audio_bytes):
    """Convert speech to text using SpeechRecognition"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name

        r = sr.Recognizer()
        with sr.AudioFile(tmp_file_path) as source:
            audio_data = r.record(source)

        text = r.recognize_google(audio_data)
        os.unlink(tmp_file_path)
        return text
    except Exception as e:
        st.error(f"Speech recognition error: {str(e)}")
        return None

def get_gemini_response(model, prompt, conversation_history, mode="conversational"):
    """Get response from Gemini AI with different modes"""
    try:
        system_prompts = {
            "conversational": "You are a helpful AI assistant. Provide natural, conversational responses.",
            "dubbing": "You are a dubbing director. Help with voice acting, timing, and dubbing suggestions.",
            "voice_cloning": "You are a voice technology expert. Provide guidance on voice cloning and synthesis.",
            "reader": "You are a professional narrator. Format text for optimal reading and provide reading suggestions."
        }

        context = system_prompts.get(mode, system_prompts["conversational"]) + "\n\n"

        if conversation_history:
            context += "Previous conversation:\n"
            for exchange in conversation_history[-5:]:
                context += f"Human: {exchange['human']}\nAI: {exchange['ai']}\n"
            context += "\nCurrent question:\n"

        full_prompt = context + prompt
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        st.error(f"Gemini AI error: {str(e)}")
        return None

def text_to_speech(text, api_key, voice_id, model_id="eleven_monolingual_v1"):
    """Convert text to speech using ElevenLabs"""
    try:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }

        data = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True
            }
        }

        response = requests.post(url, json=data, headers=headers)

        if response.status_code == 200:
            return response.content
        else:
            st.error(f"ElevenLabs API error: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        st.error(f"Text-to-speech error: {str(e)}")
        return None

def clone_voice(api_key, voice_name, audio_files):
    """Clone a voice using ElevenLabs Voice Cloning"""
    try:
        url = "https://api.elevenlabs.io/v1/voices/add"

        headers = {"xi-api-key": api_key}

        files = []
        for i, audio_file in enumerate(audio_files):
            files.append(('files', (f'sample_{i}.mp3', audio_file, 'audio/mpeg')))

        data = {
            'name': voice_name,
            'description': f'Cloned voice: {voice_name}'
        }

        response = requests.post(url, headers=headers, files=files, data=data)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Voice cloning error: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        st.error(f"Voice cloning error: {str(e)}")
        return None

def elevenlabs_reader(text, api_key, voice_id):
    """ElevenReader functionality for reading long texts"""
    try:
        # Split text into chunks for better processing
        chunks = []
        sentences = text.split('. ')
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk + sentence) < 500:  # Keep chunks under 500 chars
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.strip())

        # Generate audio for each chunk
        audio_segments = []
        progress_bar = st.progress(0)

        for i, chunk in enumerate(chunks):
            audio = text_to_speech(chunk, api_key, voice_id, "eleven_multilingual_v2")
            if audio:
                audio_segments.append(audio)
            progress_bar.progress((i + 1) / len(chunks))

        # Combine audio segments (simple concatenation)
        if audio_segments:
            combined_audio = b''.join(audio_segments)
            return combined_audio

        return None

    except Exception as e:
        st.error(f"ElevenReader error: {str(e)}")
        return None

def main():
    st.title("🎤 IIElevenLabs Remake")
    # st.markdown("### Complete Voice AI Suite powered by **Gemini 2.0 Flash** ⚡")
    # st.markdown("*Experience next-generation AI with enhanced reasoning, context awareness, and multilingual capabilities*")

    # Setup APIs
    gemini_api_key, elevenlabs_api_key = setup_apis()

    # Initialize Gemini
    gemini_model = initialize_gemini(gemini_api_key)

    # Get available voices
    if elevenlabs_api_key:
        available_voices = get_available_voices(elevenlabs_api_key)
    else:
        available_voices = {}

    # Check if APIs are configured
    if not gemini_api_key or not elevenlabs_api_key:
        st.warning("⚠️ Please configure your API keys in the sidebar to get started.")
        return

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🗣️ Conversational AI", "🎬 Dubbing", "👥 Voice Cloning", "📖 ElevenReader"])

    with tab1:
        st.header("🗣️ Conversational AI - Powered by Gemini 2.0 Flash")
        st.markdown("*Experience enhanced reasoning, multimodal understanding, and superior context awareness*")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Voice selection
            if available_voices:
                selected_voice_name = st.selectbox("Select Voice:", list(available_voices.keys()))
                voice_id = available_voices[selected_voice_name]
            else:
                voice_id = "21m00Tcm4TlvDq8ikWAM"  # Default Rachel voice

            # Audio recorder
            audio_bytes = audio_recorder(
                text="Click to record",
                recording_color="#e74c3c",
                neutral_color="#34495e",
                icon_name="microphone",
                icon_size="2x",
            )

            # Text input alternative
            text_input = st.text_area("Or type your message:", height=100)

            process_button = st.button("🚀 Process", type="primary")

        with col2:
            if st.button("🔊 Preview Voice") and available_voices:
                preview_audio = text_to_speech(
                    "Hello! This is how I sound. How can I help you today?",
                    elevenlabs_api_key,
                    voice_id
                )
                if preview_audio:
                    st.audio(preview_audio, format='audio/mpeg')

            if st.button("🗑️ Clear History"):
                st.session_state.conversation_history = []
                st.rerun()

        # Process conversational input
        if process_button and gemini_model:
            user_input = None

            if audio_bytes:
                with st.spinner("🎧 Converting speech to text..."):
                    user_input = speech_to_text(audio_bytes)
                    if user_input:
                        st.success(f"📝 Recognized: {user_input}")
            elif text_input.strip():
                user_input = text_input.strip()

            if user_input:
                with st.spinner("🤖 Thinking..."):
                    ai_response = get_gemini_response(
                        gemini_model, user_input,
                        st.session_state.conversation_history,
                        "conversational"
                    )

                if ai_response:
                    st.session_state.conversation_history.append({
                        'human': user_input,
                        'ai': ai_response
                    })

                    with st.spinner("🔊 Generating voice response..."):
                        audio_response = text_to_speech(ai_response, elevenlabs_api_key, voice_id)

                    st.success("✅ Response generated!")
                    st.markdown(f"**AI Response:** {ai_response}")

                    if audio_response:
                        st.audio(audio_response, format='audio/mpeg')

        # Conversation History
        if st.session_state.conversation_history:
            st.subheader("💬 Conversation History")
            for i, exchange in enumerate(reversed(st.session_state.conversation_history[-5:])):
                with st.expander(f"Exchange {len(st.session_state.conversation_history) - i}"):
                    st.markdown(f"**You:** {exchange['human']}")
                    st.markdown(f"**AI:** {exchange['ai']}")

    with tab2:
        st.header("🎬 Dubbing Studio")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Script Input")
            dubbing_script = st.text_area("Enter script for dubbing:", height=200)

            if available_voices:
                dubbing_voice = st.selectbox("Dubbing Voice:", list(available_voices.keys()), key="dubbing_voice")
                dubbing_voice_id = available_voices[dubbing_voice]

            timing_control = st.slider("Speech Speed", 0.5, 2.0, 1.0, 0.1)

            if st.button("🎬 Generate Dubbing", type="primary"):
                if dubbing_script:
                    with st.spinner("Generating dubbing audio..."):
                        # Get dubbing suggestions from Gemini
                        dubbing_advice = get_gemini_response(
                            gemini_model,
                            f"Provide dubbing direction for this script: {dubbing_script}",
                            [],
                            "dubbing"
                        )

                        # Generate audio with custom settings
                        dubbing_audio = text_to_speech(dubbing_script, elevenlabs_api_key, dubbing_voice_id)

                        if dubbing_audio:
                            st.success("🎬 Dubbing generated!")
                            st.audio(dubbing_audio, format='audio/mpeg')

                            if dubbing_advice:
                                st.subheader("🎭 Dubbing Direction")
                                st.markdown(dubbing_advice)

        with col2:
            st.subheader("Dubbing Tips")
            st.markdown("""
            **Professional Dubbing Guidelines:**
            - Match the emotional tone of the original
            - Consider lip-sync timing
            - Adjust pace for natural delivery
            - Use appropriate voice characteristics
            - Test with different voices for best match
            """)

    with tab3:
        st.header("👥 Voice Cloning")

        st.subheader("Clone a New Voice")

        col1, col2 = st.columns(2)

        with col1:
            voice_name = st.text_input("Voice Name:")

            st.markdown("**Upload Audio Samples** (MP3 format recommended)")
            uploaded_files = st.file_uploader(
                "Choose audio files",
                type=['mp3', 'wav', 'm4a'],
                accept_multiple_files=True
            )

            if st.button("🧬 Clone Voice", type="primary"):
                if voice_name and uploaded_files:
                    with st.spinner("Cloning voice... This may take a few minutes."):
                        audio_files = []
                        for file in uploaded_files:
                            audio_files.append(file.getvalue())

                        result = clone_voice(elevenlabs_api_key, voice_name, audio_files)

                        if result:
                            st.success(f"✅ Voice '{voice_name}' cloned successfully!")
                            st.session_state.cloned_voices.append({
                                'name': voice_name,
                                'voice_id': result.get('voice_id', ''),
                                'status': 'ready'
                            })
                        else:
                            st.error("❌ Voice cloning failed. Please try again.")
                else:
                    st.error("Please provide a voice name and upload audio samples.")

        with col2:
            st.subheader("Voice Cloning Tips")
            st.markdown("""
            **For Best Results:**
            - Upload 3-5 audio samples
            - Each sample should be 30 seconds to 2 minutes
            - Use clear, high-quality recordings
            - Include varied emotional expressions
            - Avoid background noise
            - Use consistent microphone/environment
            """)

        # Display cloned voices
        if st.session_state.cloned_voices:
            st.subheader("Your Cloned Voices")
            for voice in st.session_state.cloned_voices:
                st.markdown(f"**{voice['name']}** - Status: {voice['status']}")

    with tab4:
        st.header("📖 ElevenReader")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Text to Read")

            # Input methods
            input_method = st.radio("Input Method:", ["Type Text", "Upload Document"])

            text_to_read = ""

            if input_method == "Type Text":
                text_to_read = st.text_area("Enter text to read:", height=300)
            else:
                uploaded_file = st.file_uploader("Upload document", type=['txt', 'pdf', 'docx'])
                if uploaded_file:
                    if uploaded_file.type == "text/plain":
                        text_to_read = str(uploaded_file.read(), "utf-8")
                    else:
                        st.info("PDF and DOCX support coming soon. Please use plain text files.")

            # Reader settings
            if available_voices:
                reader_voice = st.selectbox("Reader Voice:", list(available_voices.keys()), key="reader_voice")
                reader_voice_id = available_voices[reader_voice]

            reading_speed = st.slider("Reading Speed", 0.7, 1.5, 1.0, 0.1)

            if st.button("📖 Generate Reading", type="primary"):
                if text_to_read:
                    with st.spinner("Generating reading audio... This may take a while for long texts."):
                        reader_audio = elevenlabs_reader(text_to_read, elevenlabs_api_key, reader_voice_id)

                        if reader_audio:
                            st.session_state.reader_audio = reader_audio
                            st.success("📖 Reading generated successfully!")

                            # Get reading suggestions from Gemini
                            reading_tips = get_gemini_response(
                                gemini_model,
                                f"Provide reading tips and summary for this text: {text_to_read[:500]}...",
                                [],
                                "reader"
                            )

                            if reading_tips:
                                st.subheader("📚 Reading Analysis")
                                st.markdown(reading_tips)
                else:
                    st.error("Please provide text to read.")

            # Play generated reading
            if st.session_state.reader_audio:
                st.subheader("🔊 Generated Reading")
                st.audio(st.session_state.reader_audio, format='audio/mpeg')

        with col2:
            st.subheader("ElevenReader Features")
            st.markdown("""
            **Advanced Reading Capabilities:**
            - Long-form text processing
            - Multiple voice options
            - Adjustable reading speed
            - Document upload support
            - Professional narration quality
            - Automatic text optimization

            **Supported Formats:**
            - Plain text (.txt)
            - PDF documents
            - Word documents (.docx)
            - Direct text input
            """)

            # Word count and estimated reading time
            if text_to_read:
                word_count = len(text_to_read.split())
                estimated_time = word_count / 150  # Average reading speed
                st.metric("Word Count", word_count)
                st.metric("Estimated Reading Time", f"{estimated_time:.1f} minutes")

if __name__ == "__main__":
    main()