import torch
import numpy as np

import glob
import h5py
from tqdm import tqdm

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.dataset.factory_dataset import FactoryH5Dataset
from diffusion_policy.common.random_number_generator import RandomNumberGenerator

class FactoryH5CotrainDataset(BaseImageDataset):
    """
    Dataset loader for 1019_peg_insert_bighole HDF5 format
    Compatible with victor_dp diffusion policy training
    """
    def __init__(self,
        real_h5_dir,
        sim_h5_dir,
        horizon=16,
        pad_before=1,
        pad_after=7,
        use_point_cloud=False,
        use_wrench=False,
        max_real_episodes=None,
        max_sim_episodes=None,
        max_val_episodes=None,      # evaluate on real
        alpha=0.3,      # fraction of use of real-data
        seed=42,
        preload=True
    ):

        super().__init__()

        self.real_h5_dir = real_h5_dir
        self.sim_h5_dir = sim_h5_dir
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_point_cloud = use_point_cloud
        self.use_wrench = use_wrench
        self.max_real_episodes = max_real_episodes
        self.max_sim_episodes = max_sim_episodes
        self.max_val_episodes = max_val_episodes
        self.seed = seed
        self.preload = preload
        self.alpha = alpha

        print(f"Co-training with alpha={alpha} (real={alpha}, sim={1-alpha})")

        self.real_ds = FactoryH5Dataset(
            h5_dir=real_h5_dir,
            split='training',
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            use_point_cloud=use_point_cloud,
            use_wrench=use_wrench,
            max_train_episodes=max_real_episodes,
            max_val_episodes=max_val_episodes,      # evaluate on real
            seed=seed,
            preload=preload
        )

        self.sim_ds = FactoryH5Dataset(
            h5_dir=sim_h5_dir,  # Fixed: was pointing to real_h5_dir
            split='training',
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            use_point_cloud=use_point_cloud,
            use_wrench=use_wrench,
            max_train_episodes=max_sim_episodes,
            max_val_episodes=0,      # evaluate on real
            seed=seed,
            preload=preload
        )

        # Efficient random number generator with buffered sampling
        self.rng = RandomNumberGenerator(buffer_size=10000, seed=seed)

        # Set max_val for integer sampling (will be updated in __getitem__)
        # We'll dynamically set this based on which dataset we're sampling from
        max_size = max(len(self.real_ds), len(self.sim_ds))
        if max_size > 0:
            self.rng.set_max_val(max_size)

    def get_validation_dataset(self):
        """Create validation dataset by loading from the 'validation' split directory"""
        val_set = FactoryH5Dataset(
            h5_dir=self.real_h5_dir,
            split='validation',
            horizon=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            use_point_cloud=self.use_point_cloud,
            use_wrench=self.use_wrench,
            max_train_episodes=0,
            max_val_episodes=self.max_val_episodes,
            seed=self.seed,
            preload=self.preload
        )
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        """Compute normalizer statistics using alpha-sampling strategy"""
        # Sample subset for stats with alpha-sampling
        total_sample_size = min(1000, len(self))

        # Determine how many samples from each dataset
        n_real_samples = int(total_sample_size * self.alpha)
        n_sim_samples = total_sample_size - n_real_samples

        print(f"Computing normalizer with {n_real_samples} real samples and {n_sim_samples} sim samples")

        all_acts = []
        all_obs = []

        # Sample from real dataset
        if n_real_samples > 0:
            real_sample_size = min(n_real_samples, len(self.real_ds))
            real_indices = np.random.choice(len(self.real_ds), real_sample_size, replace=False)
            for idx in tqdm(real_indices, desc="Computing normalizer (real)"):
                sample = self.real_ds[idx]
                # Take only first observation and action for normalization
                all_acts.append(sample['action'][0].numpy())
                all_obs.append(sample['obs']['robot_obs'][0].numpy())

        # Sample from sim dataset
        if n_sim_samples > 0:
            sim_sample_size = min(n_sim_samples, len(self.sim_ds))
            sim_indices = np.random.choice(len(self.sim_ds), sim_sample_size, replace=False)
            for idx in tqdm(sim_indices, desc="Computing normalizer (sim)"):
                sample = self.sim_ds[idx]
                # Take only first observation and action for normalization
                all_acts.append(sample['action'][0].numpy())
                all_obs.append(sample['obs']['robot_obs'][0].numpy())

        normalizer = LinearNormalizer()
        normalizer.fit({
            'action': np.array(all_acts),
            'robot_obs': np.array(all_obs)
        }, last_n_dims=1, mode=mode, **kwargs)

        return normalizer

    def __len__(self):
        # Combined length of both datasets
        return len(self.real_ds) + len(self.sim_ds)

    def __getitem__(self, idx):
        """Sample a sequence with probability alpha from real_ds, 1-alpha from sim_ds"""
        # Sample from real or sim dataset based on alpha probability
        # Use buffered random number generator for efficiency
        if self.rng.random() < self.alpha:
            # Sample from real dataset
            # Update max_val if needed (in case dataset sizes differ)
            if self.rng.max_val != len(self.real_ds):
                self.rng.set_max_val(len(self.real_ds))
            real_idx = self.rng.randint()
            return self.real_ds[real_idx]
        else:
            # Sample from sim dataset
            if self.rng.max_val != len(self.sim_ds):
                self.rng.set_max_val(len(self.sim_ds))
            sim_idx = self.rng.randint()
            return self.sim_ds[sim_idx]