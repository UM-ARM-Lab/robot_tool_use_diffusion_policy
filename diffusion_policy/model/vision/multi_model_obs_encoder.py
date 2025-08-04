from typing import Dict, Tuple, Union
import copy
import torch
import torch.nn as nn
import torchvision
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


class MultiModelObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            rgb_model: Union[nn.Module, Dict[str,nn.Module]],
            rgb_transform: Union[nn.Module, Dict[str,nn.Module], None]=None,
            time_series_model: Union[nn.Module, Dict[str,nn.Module], None]=None,
        ):
        """
        Assumes rgb input: B,C,H,W
        Assumes low_dim input: B,D
        """
        super().__init__()

        rgb_keys = list()
        low_dim_keys = list()
        time_series_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        # handle sharing vision backbone
        # if share_rgb_model:
            # assert isinstance(rgb_model, nn.Module)

        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)
                key_transform_map[key] = rgb_transform
                key_model_map['rgb'] = rgb_model
            elif type == 'low_dim':
                low_dim_keys.append(key)
            elif type == 'time_series':
                time_series_keys.append(key)
                key_model_map[key] = time_series_model
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        

 

        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)
        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = True
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.time_series_keys = time_series_keys
        self.key_shape_map = key_shape_map


  
    
    
    

    def forward(self, obs_dict):
        features = list()

        rgb_features_list = self.rgb_forward(obs_dict)
        features.extend(rgb_features_list)
        
        low_dim_features_list = self.lowdim_forward(obs_dict)
        features.extend(low_dim_features_list)

        time_series_features_list = self.time_series_forward(obs_dict)
        features.extend(time_series_features_list)

        # concatenate all features
        result = torch.cat(features, dim=-1)
        return result
    def rgb_forward(self, obs_dict):
        rgb_features = list()
        batch_size = None
        # process rgb input
        if self.share_rgb_model:
            # pass all rgb obs to rgb model
            imgs = list()
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]

                img = self.key_transform_map[key](img)
                imgs.append(img)
            # (N*B,C,H,W)
            imgs = torch.cat(imgs, dim=0)
            # (N*B,D)
            feature = self.key_model_map['rgb'](imgs)
            # (N,B,D)
            feature = feature.reshape(-1,batch_size,*feature.shape[1:])
            # (B,N,D)
            feature = torch.moveaxis(feature,0,1)
            # (B,N*D)
            feature = feature.reshape(batch_size,-1)
            rgb_features.append(feature)
        else:
            # run each rgb obs to independent models
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                feature = self.key_model_map[key](img)
                rgb_features.append(feature)
        
        return rgb_features

    def time_series_forward(self, obs_dict):
        """
        Only process time series input
        """
        batch_size = None
        time_series_features = list()
        for key in self.time_series_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert data.shape[1:] == self.key_shape_map[key]

            feature = self.key_model_map[key](data)
            time_series_features.append(feature)

        # concatenate all features
        # result = torch.cat(features, dim=-1)
        return time_series_features
    
    def lowdim_forward(self, obs_dict):
        """
        Only process lowdim input
        """
        batch_size = None
        low_dim_features = list()
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert data.shape[1:] == self.key_shape_map[key]
            low_dim_features.append(data)
        
        # concatenate all features
        # result = torch.cat(features, dim=-1)
        return low_dim_features

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (batch_size,) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape
