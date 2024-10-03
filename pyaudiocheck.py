import pyaudio

# Create a PyAudio object
p = pyaudio.PyAudio()

# List available audio input devices
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')

for i in range(numdevices):
    if p.get_device_info_by_index(i).get('maxInputChannels') > 0:
        print(p.get_device_info_by_index(i).get('name'))

# Close the PyAudio object
p.terminate()
