if __name__ == '__main__':
    import os
    import sys
    import torch  # Import PyTorch

    # Check for Windows and Python version
    if os.name == "nt" and (3, 8) <= sys.version_info < (3, 99):
        from torchaudio._extension.utils import _init_dll_path  # type: ignore
        _init_dll_path()

    from RealtimeSTT import AudioToTextRecorder

    # Check if GPU is available and set the device accordingly
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the AudioToTextRecorder with GPU support
    recorder = AudioToTextRecorder(
        spinner=False,
        silero_sensitivity=0.2,
        model="tiny.en",
        language="en",
        device=device,  # Specify the device here
    )

    print("Say something...")

    try:
        while True:
            print("------------------1--------------")
            print("Detected text: " + recorder.text())
    except KeyboardInterrupt:
        print("Exiting application due to keyboard interrupt")
