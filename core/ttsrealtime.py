import torch

# Load the TTS model
model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_tts', language='en', speaker='en_0')

def text_to_speech(text):
    # Ensure that the speaker is valid
    available_speakers = model.available_speakers
    print("Available speakers:", available_speakers)
    
    audio = model.apply_tts(text=text, speaker='v3_en')  # Use a valid speaker
    return audio

sentence = "This is a sentence."
audio_data = text_to_speech(sentence)
