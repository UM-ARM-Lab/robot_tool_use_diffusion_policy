from typing import Dict
import torch
import numpy as np
import copy
import zarr
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.codecs.imagecodecs_numcodecs import Jpeg2k, register_codecs, Blosc2, Jpeg

register_codecs()

class VictorLowdimImageDataset(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=0,
            val_ratio=0.0,
            max_train_episodes=None,
            act_keys=['robot_act'],
            obs_keys=['right_joint_positions','gripper_states',],
            image_keys=['image'],
            ):
        
        super().__init__()

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=act_keys + obs_keys + image_keys,)
         

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask
            )
        self.keys = act_keys + obs_keys
        self.act_keys = act_keys
        self.obs_keys = obs_keys
        self.image_keys = image_keys
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        # obs_dic = {}
        # for obj_key in self.obs_keys:
            # obs_dic[obj_key] = self.replay_buffer[obj_key]
        # obs_np = np.concatenate([np.array(self.replay_buffer[key]) for key in self.obs_keys], axis=-1).astype(np.float32)

        data_dic = {
            key : np.array(self.replay_buffer[key]) for key in self.obs_keys
        }
        data_dic['action'] = np.array(self.replay_buffer[self.act_keys[0]])  # Assuming single action key
 
        normalizer = LinearNormalizer()
        normalizer.fit(data=data_dic, last_n_dims=1, mode=mode, **kwargs)
        normalizer[self.image_keys[0]] = get_image_range_normalizer()
        return normalizer


    def __len__(self) -> int:
        return len(self.sampler)


    def _sample_to_data(self, sample):
        # obs_np = np.concatenate([sample[key].astype(np.float32) for key in self.obs_keys], axis=-1).astype(np.float32)

        obs_dic = { key: sample[key].astype(np.float32) for key in self.obs_keys}
        image = np.moveaxis(sample[self.image_keys[0]],-1,1)/255
        obs_dic[self.image_keys[0]] = image

        action_np = sample[self.act_keys[0]].astype(np.float32)

        data = {
            'obs': obs_dic,  # 
            'action': action_np # T, 2
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


def test():
    pass
 
 
if __name__ == "__main__":
    import os
    test()