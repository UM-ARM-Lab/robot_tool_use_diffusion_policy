import torch
import torchvision
from typing import Dict, Tuple, Union
import torch.nn as nn
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer

 
def get_audio_model(name):
    if name == 'ast':
        from diffusion_policy.model.audio.lstm_encoder import LSTMEncoder
        return LSTMEncoder
    elif name == 'conv1d':
        from diffusion_policy.model.audio.conv1d_encoder import Conv1dEncoder
        return Conv1dEncoder
    else:
        raise ValueError(f"Unknown audio model name: {name}") 