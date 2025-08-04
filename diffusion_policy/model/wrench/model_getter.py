import torch
import torchvision
from typing import Dict, Tuple, Union
import torch.nn as nn
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer

from diffusion_policy.model.wrench.LSTM_encoder import LSTMEncoder


def get_wrench_encoder(name,**kwargs):
    if name == "lstm":
        return LSTMEncoder(**kwargs)
    

