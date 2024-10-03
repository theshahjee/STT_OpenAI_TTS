# main.py

from RealtimeSTT import AudioToTextRecorder
import torch
from openai_integration import test_openai

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

    print("Say something...")

    try:
        while True:
            # Capture user speech and convert to text
            user_speech = recorder.text()
            print("UserSays: " + user_speech)
            
            # Send the text to OpenAI for processing
            ai_response = test_openai(user_speech)
            print("Doctor's Response: " + ai_response)

    except KeyboardInterrupt:
        print("Exiting application due to keyboard interrupt")
