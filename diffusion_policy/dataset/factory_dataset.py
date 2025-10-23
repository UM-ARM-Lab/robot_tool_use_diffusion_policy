import torch
import numpy as np

import glob
import h5py
from tqdm import tqdm

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset

class FactoryH5Dataset(BaseImageDataset):
    """
    Dataset loader for 1019_peg_insert_bighole HDF5 format
    Compatible with victor_dp diffusion policy training
    """
    def __init__(self,
        h5_dir,
        split='training',
        horizon=16,
        pad_before=1,
        pad_after=7,
        use_point_cloud=False,
        use_wrench=False,
        max_train_episodes=None,
        max_val_episodes=None,
        seed=42,
        preload=True
    ):

        super().__init__()

        # Load all h5 file paths from the specified split directory
        h5_paths = sorted(glob.glob(f"{h5_dir}/{split}/*.h5"))

        if len(h5_paths) == 0:
            raise ValueError(f"No H5 files found in {h5_dir}/{split}/")

        # Determine number of episodes per file and episode length from first file
        with h5py.File(h5_paths[0], 'r') as f:
            self.episodes_per_file = f['action'].shape[0]
            self.episode_length = f['action'].shape[1]

        print(f"Detected {self.episodes_per_file} episodes per file, {self.episode_length} timesteps per episode")

        # Build episode index: (file_path, ep_idx_in_file)
        all_episodes = []
        for h5_path in h5_paths:
            for ep_in_file in range(self.episodes_per_file):
                all_episodes.append((h5_path, ep_in_file))

        print(f"Total available episodes: {len(all_episodes)}")

        # Randomly sample episodes if max_episodes is specified
        max_episodes = max_train_episodes if split == 'training' else max_val_episodes
        if max_episodes is not None and max_episodes < len(all_episodes):
            rng = np.random.RandomState(seed)
            sampled_indices = rng.choice(len(all_episodes), size=max_episodes, replace=False)
            self.episodes = [all_episodes[i] for i in sampled_indices]
            print(f"Randomly sampled {len(self.episodes)} episodes from {len(all_episodes)} available")
        else:
            self.episodes = all_episodes
            print(f"Using all {len(self.episodes)} episodes")

        self.h5_dir = h5_dir
        self.split = split
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_point_cloud = use_point_cloud
        self.use_wrench = use_wrench
        self.max_train_episodes = max_train_episodes
        self.max_val_episodes = max_val_episodes
        self.seed = seed
        self.preload = preload

        # Pre-load all data into memory if requested
        self.preloaded_data = None
        if self.preload:
            print(f"Pre-loading all {len(self.episodes)} episodes into memory...")
            self.preloaded_data = {}
            for ep_global_idx, (h5_path, ep_idx) in enumerate(tqdm(self.episodes, desc="Loading episodes")):
                # Load episode data
                with h5py.File(h5_path, 'r') as f:
                    data = {
                        'robot_act': f['action'][ep_idx].astype('f4'),
                        'robot_obs': f['low_dim_state'][ep_idx].astype('f4'),
                    }
                    if self.use_wrench:
                        data['wrench'] = f['wrench'][ep_idx].astype('f4')
                        data['tool_pose'] = f['tool_pose'][ep_idx].astype('f4')
                    if self.use_point_cloud:
                        data['partial_pc'] = f['partial_pc'][ep_idx].astype('f4')

                # Store in memory using global episode index
                self.preloaded_data[ep_global_idx] = data
            print(f"Pre-loading complete! All data loaded into memory.")

        # Cache for frequently accessed h5 files (only used if preload=False)
        self.h5_cache = {}
        self.max_cache_size = 10

    def get_validation_dataset(self):
        """Create validation dataset by loading from the 'validation' split directory"""
        val_set = FactoryH5Dataset(
            h5_dir=self.h5_dir,
            split='validation',
            horizon=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            use_point_cloud=self.use_point_cloud,
            use_wrench=self.use_wrench,
            max_train_episodes=self.max_train_episodes,
            max_val_episodes=self.max_val_episodes,
            seed=self.seed,
            preload=self.preload
        )
        return val_set

    def _load_episode(self, ep_global_idx, h5_path=None, ep_idx=None):
        """Load single episode from memory or h5 file with caching

        Args:
            ep_global_idx: Global episode index in self.episodes list
            h5_path: H5 file path (only used if not preloaded)
            ep_idx: Episode index within H5 file (only used if not preloaded)
        """
        # If data is preloaded, return from memory
        if self.preload:
            return self.preloaded_data[ep_global_idx]

        # Otherwise, load from H5 file with caching
        if h5_path not in self.h5_cache:
            if len(self.h5_cache) >= self.max_cache_size:
                # Remove oldest
                self.h5_cache.pop(next(iter(self.h5_cache)))
            self.h5_cache[h5_path] = h5py.File(h5_path, 'r')

        f = self.h5_cache[h5_path]

        data = {
            'robot_act': f['action'][ep_idx].astype('f4'),
            'robot_obs': f['low_dim_state'][ep_idx].astype('f4'),
        }

        if self.use_wrench:
            data['wrench'] = f['wrench'][ep_idx].astype('f4')
            data['tool_pose'] = f['tool_pose'][ep_idx].astype('f4')

        if self.use_point_cloud:
            data['partial_pc'] = f['partial_pc'][ep_idx].astype('f4')

        return data

    def get_normalizer(self, mode='limits', **kwargs):
        """Compute normalizer statistics"""
        # Sample subset for stats
        sample_size = min(1000, len(self.episodes))
        sample_indices = np.random.choice(len(self.episodes), sample_size, replace=False)

        all_acts = []
        all_obs = []

        for idx in tqdm(sample_indices, desc="Computing normalizer"):
            h5_path, ep_idx = self.episodes[idx]
            data = self._load_episode(ep_global_idx=idx, h5_path=h5_path, ep_idx=ep_idx)
            all_acts.append(data['robot_act'])
            all_obs.append(data['robot_obs'])

        normalizer = LinearNormalizer()
        normalizer.fit({
            'action': np.array(all_acts),
            'robot_obs': np.array(all_obs)
        }, last_n_dims=1, mode=mode, **kwargs)

        return normalizer

    def __len__(self):
        # Number of valid windows per episode
        return len(self.episodes) * (self.episode_length - self.horizon + 1)

    def __getitem__(self, idx):
        """Sample a sequence"""
        ep_len = self.episode_length - self.horizon + 1
        ep_idx = idx // ep_len
        start_t = idx % ep_len
        end_t = start_t + self.horizon

        h5_path, ep_in_file = self.episodes[ep_idx]
        data = self._load_episode(ep_global_idx=ep_idx, h5_path=h5_path, ep_idx=ep_in_file)

        # Extract sequence
        result = {
            'obs': {
                'robot_obs': torch.from_numpy(data['robot_obs'][start_t:end_t])
            },
            'action': torch.from_numpy(data['robot_act'][start_t:end_t])
        }

        if self.use_wrench:
            result['obs']['wrench'] = torch.from_numpy(data['wrench'][start_t:end_t])
            result['obs']['tool_pose'] = torch.from_numpy(data['tool_pose'][start_t:end_t])

        if self.use_point_cloud:
            pc = data['partial_pc'][start_t:end_t]  # (T, 1024, 3)
            result['obs']['point_cloud'] = torch.from_numpy(pc)

        return result