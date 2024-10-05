from TTS.api import TTS
import torch

def is_gpu_available():
    return torch.cuda.is_available()

# Initialize TTS model (use GPU if available)
tts_model = TTS(model_name="tts_models/en/vctk/vits", gpu=is_gpu_available())

def list_speakers():
    """List all available speakers in the TTS model."""
    return tts_model.speakers

def generate_audio(text, speaker=None, file_path="output.wav"):
    """Generate speech audio from text and save it to a file."""
    if speaker is None:
        speaker = tts_model.speakers[0]
    tts_model.tts_to_file(text=text, speaker=speaker, file_path=file_path)
