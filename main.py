from audio_utlis import record_audio, transcribe_audio
from llm import generate_response
from tts_utils import list_speakers, generate_audio
import os


def run_conversation():
    history = []

    # List available speakers
    speakers = list_speakers()
    print("Available speakers:", speakers)

    # Loop to continuously listen for input and respond
    while True:
        # Record user input
        filename = record_audio()

        # Transcribe audio
        user_text = transcribe_audio(filename)

        # Check for exit condition
        if user_text.lower() in ["exit", "quit", "stop"]:
            print("Exiting the conversation.")
            break

        print(f"User: {user_text}")
        history.append({'role': 'user', 'content': user_text})

        # Get assistant response from GPT-3
        system_prompt = {'role': 'system', 'content': 'Answer with polite sarcasm.'}
        assistant_response = generate_response([system_prompt] + history)
        print(f"Assistant: {assistant_response}")

        # Add assistant's response to the conversation history
        history.append({'role': 'assistant', 'content': assistant_response})

        # Generate audio response
        generate_audio(text=assistant_response)

        # Play the generated response (assistant's audio only)
        os.system("aplay output.wav")


if __name__ == "__main__":
    run_conversation()