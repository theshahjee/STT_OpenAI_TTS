import os
from dotenv import load_dotenv
import torch
from TTS.api import TTS
from faster_whisper import WhisperModel
import sounddevice as sd
from scipy.io.wavfile import write
from openai import OpenAI

# Load environment variables from the .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Function to check if GPU is available
def is_gpu_available():
    return torch.cuda.is_available()


# Initialize TTS model (use GPU if available)
tts_model = TTS(model_name="tts_models/en/vctk/vits", gpu=is_gpu_available())

# Initialize Whisper model (use GPU if available)
whisper_model = WhisperModel("tiny", device="cuda" if is_gpu_available() else "cpu")


# Function to record audio
def record_audio(duration=5, filename="temp.wav", sample_rate=44100):
    print(f"Recording for {duration} seconds at {sample_rate} Hz on device {sd.default.device}...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # Wait until the recording is finished
    write(filename, sample_rate, audio_data)
    print("Recording complete!")


# System prompt for OpenAI GPT-3
system_prompt_message = {
    'role': 'system',
    'content': 'Answer precise and short with the polite sarcasm of a butler.'
}


# Function to generate responses using OpenAI GPT-3
def generate_response(messages):
    """Generate assistant's response using OpenAI GPT-3."""
    for chunk in client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, stream=True):
        text_chunk = chunk["choices"][0]["delta"].get("content")
        if text_chunk:
            yield text_chunk


# Main conversation loop to handle speech-to-text, GPT-3 responses, and text-to-speech
def run_conversation():
    history = []

    # List available speakers in the TTS model
    print("Available speakers:", tts_model.speakers)

    # Use the first available speaker (you can choose a different one)
    speaker = tts_model.speakers[0]

    # Record user input from microphone
    record_audio()  # Record a 5-second audio file as temp.wav

    print("Playing the recorded audio from temp.wav...")
    os.system("aplay temp.wav")  # Play the recorded audio

    print("Transcribing audio...")
    audio_data, _ = whisper_model.transcribe("temp.wav")  # The transcribe method returns a tuple

    # Extract text from the transcription result
    user_text = ''.join([segment.text for segment in audio_data]).strip()

    if not user_text:
        return

    print(f"User: {user_text}")
    history.append({'role': 'user', 'content': user_text})

    # Get assistant response from OpenAI
    assistant_response = ''.join(list(generate_response([system_prompt_message] + history[-10:])))
    print(f"Assistant: {assistant_response}")

    # Convert the GPT-3 text response to audio using the TTS model with the selected speaker
    tts_model.tts_to_file(text=assistant_response, speaker=speaker, file_path="output.wav")

    # Play the generated audio response
    os.system("aplay output.wav")


if __name__ == "__main__":
    run_conversation()
