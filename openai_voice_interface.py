import os
import torch
import sounddevice as sd
import numpy as np
import queue
import threading
from TTS.api import TTS
from faster_whisper import WhisperModel
from openai import OpenAI
from dotenv import load_dotenv
from scipy.io.wavfile import write

# Load environment variables from the .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to check if GPU is available
def is_gpu_available():
    return torch.cuda.is_available()

# Initialize a better TTS model (use GPU if available)
tts_model = TTS(model_name="tts_models/en/vctk/vits", gpu=is_gpu_available())

# Initialize Whisper model (use GPU if available)
whisper_model = WhisperModel("small.en", device="cuda" if is_gpu_available() else "cpu")

# Queue to hold audio data for real-time processing
audio_queue = queue.Queue()

# Callback function to collect audio data in chunks
def callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

# Record audio in real-time and dynamically adjust for valid sample rate
def record_audio_stream():
    # Check available devices and sample rates
    device_info = sd.query_devices()
    print("Available devices and sample rates:")
    print(device_info)

    # Identify the microphone (replace with your specific device index if needed)
    device_index = 17  # You may need to adjust this based on your microphone
    sample_rate = 44100  # Set a commonly supported sample rate (adjust if necessary)

    # Adjust based on available sample rates if necessary
    try:
        with sd.InputStream(callback=callback, device=device_index, channels=1, samplerate=sample_rate, blocksize=1024):
            while True:
                sd.sleep(1000)
    except sd.PortAudioError as e:
        print(f"Error initializing input stream: {e}")
        print("Please ensure that the sample rate and input device are correct.")

# Transcribe audio from stream using Whisper
def transcribe_audio_stream():
    while True:
        if not audio_queue.empty():
            audio_chunk = audio_queue.get()
            # Save audio chunk as temporary WAV file
            write("temp.wav", 44100, np.array(audio_chunk, dtype=np.float32))

            # Play the recorded audio to verify it was captured correctly
            print("Playing the recorded audio from temp.wav...")
            os.system("aplay temp.wav")

            # Transcribe speech from WAV file
            print("Transcribing audio...")
            segments, _ = whisper_model.transcribe("temp.wav")
            transcribed_text = ''.join([segment.text for segment in segments])
            print(f"Transcribed text: {transcribed_text}")
            return transcribed_text.strip()

# Function to interact with OpenAI GPT-3 and generate responses
def generate_response(messages):
    """Generate assistant's response using OpenAI GPT-3."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response.choices[0].message.content.strip()

# Real-time speech-to-text and response function
def run_conversation():
    history = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    while True:
        # Transcribe audio in real-time
        user_text = transcribe_audio_stream()

        if user_text:
            print(f"User: {user_text}")

            # Add user input to chat history
            history.append({"role": "user", "content": user_text})

            # Generate assistant response based on conversation history
            assistant_response = generate_response(history)

            print(f"Assistant: {assistant_response}")

            # Add assistant response to chat history
            history.append({"role": "assistant", "content": assistant_response})

            # Convert the GPT-3 text response to audio using the TTS model
            tts_model.tts_to_file(text=assistant_response, speaker=None, file_path="output.wav")

            # Play the audio response
            os.system("aplay output.wav")

# Run audio recording and processing in separate threads
if __name__ == "__main__":
    # Start audio recording in a separate thread
    threading.Thread(target=record_audio_stream, daemon=True).start()

    # Start real-time transcription and interaction
    run_conversation()
