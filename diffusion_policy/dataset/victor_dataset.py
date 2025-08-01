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

class VictorDataset(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=0,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        
        super().__init__()
        self.store = zarr.MemoryStore()

        # compressor_map = {
        #     "robot_act" : Blosc2(),
        #     "robot_obs" : Blosc2(),
        #     "image"     : Jpeg2k(level=50)
        # }

        # chunk_map = {
        #     "robot_act" : (100, 11),
        #     "robot_obs" : (100, 21),
        #     "image"     : (10, 512, 512, 4)
        # }

        # self.replay_buffer = ReplayBuffer.copy_from_path(
        #     zarr_path, keys=["robot_act", "robot_obs", 'image'],
        #     # chunks=chunk_map,
        #     store=self.store,)
        #     # compressors=compressor_map)
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=["robot_act", "robot_obs", 'image']
        )
        print(self.replay_buffer.keys())
        print(f"Loaded replay buffer with {self.replay_buffer}")

        

        # Get the mask for validation episodes
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        print("VALIDATION MASK:", val_mask)
        train_mask = ~val_mask 
        # train_mask = downsample_mask(
        #     mask=train_mask, 
        #     max_n=max_train_episodes, 
        #     seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        print("HORIZON:", self.horizon)

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
        data = {
            'action': self.replay_buffer['robot_act'],
            'robot_obs'   : self.replay_buffer['robot_obs']
            # NOTE for future reference: original grabs the first 2 columns, which correspond to agent_x, agent_y
            # 'wrench': self.replay_buffer['wrench'],#[...,:2]
            # 'gripper_status': self.replay_buffer['gripper_status']
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)   # TODO unsure about keeping last_n_dims the same
        normalizer['image'] = get_image_range_normalizer()  # TODO unsure?
        # print(normalizer)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # agent_pos = sample['state'][:,:].astype(np.float32) # (agent_posx2, block_posex3)
        # motion_status = sample['action'].astype(np.float32)
        # wrench = sample['wrench'].astype(np.float32)
        # gripper_status = sample['gripper_status'].astype(np.float32)
        image = np.moveaxis(sample['image'],-1,1)/255   # unsure what this does, but everything breaks without it
        # image = sample['image'] / 255
        robot_act = sample['robot_act'].astype(np.float32)
        robot_obs = sample['robot_obs'].astype(np.float32)
    
        data = {
            'obs': {
                'image': image, # T, 3, 96, 96
                'robot_obs'  : robot_obs
                # 'action': action, # T, 7
                # 'gripper_status': gripper_status, # T, 3 # TODO ???
                # 'wrench': wrench # T, 6
            },
            'action': robot_act # T, 7    # TODO should probably include the gripper status too ? and the gripper?
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


def test():
    # zarr_path = os.path.expanduser('~/robot_tool_use_diffusion_policy/data/victor/victor_data.zarr')
    zarr_path = os.path.expanduser('~/robot_tool_use_diffusion_policy/data/victor/victor_data_07_31_no_plat.zarr')
    # zarr_path = os.path.expanduser('~/robot_tool_use_diffusion_policy/data/victor/victor_data.zarr/ds_processed.zarr.zip')
    dataset = VictorDataset(zarr_path, horizon=1)
    print(dataset.replay_buffer.data)
    print(dataset.__getitem__(0))
    print(dataset.replay_buffer.episode_ends[:])
    print(dataset.replay_buffer.n_episodes)
    dataset.get_normalizer()
    # from matplotlib import pyplot as plt
    # normalizer = dataset.get_normalizer()
    # nactions = normalizer['action'].normalize(dataset.replay_buffer['action'])
    # diff = np.diff(nactions, axis=0)
    # dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)

if __name__ == "__main__":
    import os
    test()