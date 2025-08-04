import torch
import torchvision
from typing import Dict, Tuple, Union
import torch.nn as nn
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer

def get_resnet(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    # load r3m weights
    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)

    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    return resnet

def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m
    r3m.device = 'cpu'
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to('cpu')
    return resnet_model



def get_rgb_transform(image_shape,
                      resize_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            random_crop: bool=True,
            use_group_norm: bool=False,
            share_rgb_model: bool=False,
            imagenet_norm: bool=False):
    """
    Returns a torchvision transform for RGB images.
    """
    input_shape = image_shape
    this_resizer = nn.Identity()
    if resize_shape is not None:
        
        h, w = resize_shape
        this_resizer = torchvision.transforms.Resize(
            size=(h,w)
        )
        input_shape = (image_shape[0],h,w)

    # configure randomizer
    this_randomizer = nn.Identity()
    if crop_shape is not None:
        h, w = crop_shape
        if random_crop:
            this_randomizer = CropRandomizer(
                input_shape=input_shape,
                crop_height=h,
                crop_width=w,
                num_crops=1,
                pos_enc=False
            )
        else:
            this_normalizer = torchvision.transforms.CenterCrop(
                size=(h,w)
            )
    # configure normalizer
    this_normalizer = nn.Identity()
    if imagenet_norm:
        this_normalizer = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    this_transform = nn.Sequential(this_resizer, this_randomizer, this_normalizer)
    return  this_transform