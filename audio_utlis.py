import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
import torch

def is_gpu_available():
    return torch.cuda.is_available()

# Initialize Whisper model (use GPU if available)
whisper_model = WhisperModel("tiny", device="cuda" if is_gpu_available() else "cpu")

def record_audio(filename="temp.wav", duration=5, sample_rate=44100):
    """Records audio for a given duration and saves to a file."""
    print(f"Recording for {duration} seconds...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # Wait until recording is done
    write(filename, sample_rate, audio_data)
    print("Recording complete!")
    return filename

def transcribe_audio(filename="temp.wav"):
    """Transcribes audio from a file and returns the text."""
    audio_data, _ = whisper_model.transcribe(filename)
    return ''.join([segment.text for segment in audio_data]).strip()
