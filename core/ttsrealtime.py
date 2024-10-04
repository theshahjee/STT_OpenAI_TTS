import torch
import soundfile as sf


language = 'en'
speaker = 'lj_16khz'
device = torch.device('cuda')  


model, symbols, sample_rate, example_text, apply_tts = torch.hub.load(
    repo_or_dir='snakers4/silero-models', 
    model='silero_tts', 
    language=language, 
    speaker=speaker
)


model = model.to(device)


text = "Hello Motherfucker, how are you today? Thank you for reaching out. It seems like there might have been a mix-up in your message. If you have any health-related queries or concerns, please feel free to share them with me. It's important to provide details such as your name, age, specific symptoms, and how long you've been experiencing the issue so I can offer you the most appropriate medical advice."

audio = apply_tts(
    texts=[text], 
    model=model,  
    sample_rate=sample_rate,  
    symbols=symbols, 
    device=device  
)


# sf.write('output.wav', audio[0], sample_rate)  

# print("TTS audio saved as 'output.wav'")
