import os
import torch

device = torch.device('cuda')
torch.set_num_threads(4)
local_file = 'model.pt'

# Check if the model file exists locally, download if not
if not os.path.isfile(local_file):
    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/en/v2_en.pt',
                                   local_file)  # English model link

# Load the model
model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
model.to(device)

# English text examples
example_batch = ['In the depths of the tundra, otters gather cedar kernels in buckets.',
                 'Cats are like liquid!',
                 'Mom washed Mila with soap.']

sample_rate = 16000

# Adjust the speed, for fast set speed < 1.0 (e.g., 0.8 for faster speech)
audio_paths = model.save_wav(texts=example_batch,
                             sample_rate=sample_rate,
                             speed=0.8)  # Fast speed

print(audio_paths)
