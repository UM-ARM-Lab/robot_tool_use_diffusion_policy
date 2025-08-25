"""
Enhanced Victor dataset class for multimodal training with joint angles, images, and force/wrench data.

This dataset handles:
- Joint angle observations (7 DoF)  
- Gripper state observations (4 dims)
- Force/torque wrench data (6 dims)
- RGB camera images
- Optional wrench data augmentation
"""
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
from diffusion_policy.augmentation.wrench_data_augmentation import fft_augment

register_codecs()

class VictorMultiModalDataset(BaseImageDataset):
    def __init__(self,
        zarr_path, 
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=0,
        val_ratio=0.0,
        max_train_episodes=None,
        act_keys=['robot_act'],
        obs_keys=['right_joint_positions', 'gripper_states', 'wrench_data'],
        image_keys=['image'],
        wrench_augmentation=None,
):
        """
        Args:
            zarr_path: Path to dataset
            horizon: Sequence length
            pad_before/pad_after: Padding
            seed: Random seed
            val_ratio: Validation split ratio
            max_train_episodes: Maximum training episodes
            act_keys: Action data keys
            obs_keys: Low-dim observation keys (joint angles, gripper, wrench)
            image_keys: Image observation keys
            wrench_augmentation: Dict with wrench augmentation params
        """
        super().__init__()

        # Load all required keys
        all_keys = act_keys + obs_keys + image_keys
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=all_keys)
        
        print(f"Loaded multimodal dataset with keys: {all_keys}")
        print(f"Dataset contains {self.replay_buffer.n_episodes} episodes")

        # Validation split
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
        
        # Store configuration
        self.act_keys = act_keys
        self.obs_keys = obs_keys
        self.image_keys = image_keys
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.wrench_augmentation = wrench_augmentation or {}

        # Check if wrench data is available
        self.has_wrench = 'wrench_data' in obs_keys
        if self.has_wrench:
            print("✓ Wrench/force data enabled for training")
        
        print(f"Training on {train_mask.sum()} episodes, validating on {val_mask.sum()} episodes")

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
        """
        Get normalizer for all modalities
        """
        # Create data dict for normalization
        data_dic = {}
        
        # Add low-dim observations
        for key in self.obs_keys:
            data_dic[key] = np.array(self.replay_buffer[key])
        
        # Add action data    
        data_dic['action'] = np.array(self.replay_buffer[self.act_keys[0]])
        
        # Create normalizer
        normalizer = LinearNormalizer()
        normalizer.fit(data=data_dic, last_n_dims=1, mode=mode, **kwargs)
        
        # Add image normalizer
        if self.image_keys:
            normalizer[self.image_keys[0]] = get_image_range_normalizer()
            
        return normalizer

    def _augment_wrench_data(self, wrench_data):
        """
        Apply FFT-based augmentation to wrench data
        """
        if not self.wrench_augmentation.get('enabled', False):
            return wrench_data
            
        # Apply augmentation
        augmented = fft_augment(
            wrench_data,
            noise_scale=self.wrench_augmentation.get('noise_scale', 0.02),
            mode=self.wrench_augmentation.get('mode', 'topk'),
            k=self.wrench_augmentation.get('k', 5),
            sr=self.wrench_augmentation.get('sampling_rate', 200)
        )
        
        return augmented

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        """
        Convert sample to training format
        """
        obs_dict = {}
        
        # Process low-dim observations
        for key in self.obs_keys:
            data = sample[key].astype(np.float32)
            
            # Apply wrench augmentation during training
            if key == 'wrench_data' and self.has_wrench:
                # Only augment during training (not validation)
                if hasattr(self, '_is_training') and self._is_training:
                    data = self._augment_wrench_data(data)
                    
            obs_dict[key] = data
        
        # Process images
        for key in self.image_keys:
            # Standard image preprocessing: HWC -> CHW, normalize to [0,1]
            image = np.moveaxis(sample[key], -1, 1) / 255.0
            obs_dict[key] = image
            
        # Process actions
        action_np = sample[self.act_keys[0]].astype(np.float32)

        data = {
            'obs': obs_dict,
            'action': action_np
        }
        
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Set training flag for augmentation
        self._is_training = True
        
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


def test():
    """
    Test the multimodal dataset
    """
    import os
    
    # Example usage
    zarr_path = os.path.expanduser('~/data/victor/victor_multimodal_data.zarr')
    
    dataset = VictorMultiModalDataset(
        zarr_path=zarr_path,
        horizon=16,
        obs_keys=['right_joint_positions', 'gripper_states', 'wrench_data'],
        image_keys=['image'],
        wrench_augmentation={
            'enabled': True,
            'noise_scale': 0.02,
            'mode': 'topk',
            'k': 5
        }
    )
    
    print("Dataset shape info:")
    print(f"Length: {len(dataset)}")
    
    # Test sample
    sample = dataset[0]
    print("\nSample structure:")
    for key, value in sample.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for subkey, subvalue in value.items():
                print(f"  {subkey}: {subvalue.shape}")
        else:
            print(f"{key}: {value.shape}")
            
    # Test normalizer
    normalizer = dataset.get_normalizer()
    print(f"\nNormalizer keys: {list(normalizer.keys())}")


if __name__ == "__main__":
    test()
