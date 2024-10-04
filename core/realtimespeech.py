if __name__ == '__main__':
    import os
    import sys
    import torch

    if os.name == "nt" and (3, 8) <= sys.version_info < (3, 99):
        from torchaudio._extension.utils import _init_dll_path  # type: ignore
        _init_dll_path()

    from RealtimeSTT import AudioToTextRecorder

    device = "cuda" if torch.cuda.is_available() else "cpu"

    recorder = AudioToTextRecorder(
        spinner=False,
        silero_sensitivity=0.5,
        model="tiny.en",
        language="en",
        device=device,
    )
    
    print("Say something...")

    try:
        while True:
            print("UserSays: " + recorder.text())
    except KeyboardInterrupt:
        print("Exiting application due to keyboard interrupt")
