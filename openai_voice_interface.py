import os
from dotenv import load_dotenv
import torch
from TTS.api import TTS
from faster_whisper import WhisperModel
from openai import OpenAI  # New import based on the refactor
import sounddevice as sd
from scipy.io.wavfile import write

# Load environment variables from the .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Use the client object

# Function to check if GPU is available
def is_gpu_available():
    return torch.cuda.is_available()

# Initialize TTS model (use GPU if available)
tts_model = TTS(model_name="tts_models/en/ljspeech/glow-tts", gpu=is_gpu_available())

# Initialize Whisper model (use GPU if available)
whisper_model = WhisperModel("small", device="cuda" if is_gpu_available() else "cpu")

# Function to record audio
def record_audio(duration=5, filename="input.wav", sample_rate=44100):
    # Set the device to the correct microphone (ALC897 Analog)
    sd.default.device = 0  # Index of 'HDA Intel PCH: ALC897 Analog'

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
    response = client.chat.completions.create(  # Updated to use the client
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response.choices[0].message.content.strip()  # Properly access the response content

# Main loop to handle speech-to-text, GPT-3 responses, and text-to-speech
def main():
    """Main loop for interaction."""
    history = []

    # Record user input from microphone
    record_audio()  # Record a 5-second audio file as input.wav

    while True:
        # Transcribe user input from the recorded audio file
        audio_data, _ = whisper_model.transcribe("input.wav")  # Extracting the transcription and ignoring metadata
        user_text = ''.join([segment.text for segment in audio_data]).strip()  # Extract text from segments

        if not user_text:
            continue

        print(f'>>> {user_text}\n<<< ', end="", flush=True)
        history.append({'role': 'user', 'content': user_text})

        # Get assistant response from OpenAI
        assistant_response = generate_response([system_prompt_message] + history[-10:])

        # Convert the GPT-3 text response to audio using the TTS model
        tts_model.tts_to_file(text=assistant_response, file_path="output.wav")
        history.append({'role': 'assistant', 'content': assistant_response})

        # Play the audio response
        os.system("aplay output.wav")


if __name__ == "__main__":
    main()
