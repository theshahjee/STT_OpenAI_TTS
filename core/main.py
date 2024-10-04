# main.py

from RealtimeSTT import AudioToTextRecorder
import torch
from openai_integration import response_openai
import soundfile as sf
import os

# TTS setup (Silero model)
def initialize_tts():
    language = 'en'
    speaker = 'lj_16khz'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Silero TTS model
    model, symbols, sample_rate, example_text, apply_tts = torch.hub.load(
        repo_or_dir='snakers4/silero-models', 
        model='silero_tts', 
        language=language, 
        speaker=speaker
    )
    model = model.to(device)
    return model, symbols, sample_rate, apply_tts, device

# Function to convert text to speech
def text_to_speech(text, model, symbols, sample_rate, apply_tts, device, output_file='output.wav'):
    audio = apply_tts(
        texts=[text],  # List of text inputs to convert to speech
        model=model,  # The TTS model
        sample_rate=sample_rate,  # The sample rate for output audio
        symbols=symbols,  # TTS symbols
        device=device  # The device to run the model on
    )

    # Save the synthesized speech to a WAV file
    sf.write(output_file, audio[0], sample_rate)
    print(f"TTS audio saved as '{output_file}'")
    os.system(f"aplay {output_file}" if os.name == 'posix' else f"start {output_file}")  # Play the audio on Linux or Windows

if __name__ == '__main__':
    # Initialize real-time speech-to-text
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    recorder = AudioToTextRecorder(
        spinner=False,
        silero_sensitivity=0.2,
        model="tiny.en",
        language="en",
        device=device,
    )

    # Initialize TTS (Silero)
    tts_model, tts_symbols, tts_sample_rate, apply_tts, tts_device = initialize_tts()

    # Initialize empty chat history
    chat_history = []

    print("Say something...")

    try:
        while True:
            # Capture user speech and convert to text
            user_speech = recorder.text()
            print("UserSays: " + user_speech)
            
            # Send the text to OpenAI for processing, including chat history
            ai_response, chat_history = response_openai(user_speech, chat_history)
            print("Doctor's Response: " + ai_response)

            # Convert the AI response to speech
            text_to_speech(ai_response, tts_model, tts_symbols, tts_sample_rate, apply_tts, tts_device)

    except KeyboardInterrupt:
        print("Exiting application due to keyboard interrupt")
