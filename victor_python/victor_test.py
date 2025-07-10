import time
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
import zarr
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import matplotlib.pyplot as plt

from diffusion_policy.common.victor_accumulator import ObsAccumulator
from diffusion_policy.codecs.imagecodecs_numcodecs import (
    register_codecs,
    Jpeg2k
)
register_codecs()


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)

    output_dir = "data/victor_eval_output"
    # if os.path.exists(output_dir):
    #     click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    

    # load checkpoint
    # state only - low dim - validation
    payload = torch.load(open("data/outputs/2025.07.10/11.50.13_victor_diffusion_state_victor_diff/checkpoints/latest.ckpt", "rb"), pickle_module=dill)

    cfg = payload['cfg']
    cfg.policy.num_inference_steps = 16

    use_images = "image" in cfg['shape_meta']

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model # type: ignore
    if cfg.training.use_ema:
        policy = workspace.ema_model # type: ignore
    device = policy.device

    device = torch.device(device)
    policy.to(device)

    zf = zarr.open("data/victor/victor_data_07_10.zarr", mode='r') 
    
    vic_acc = ObsAccumulator(8)

    for i in range(696):
        print('iter:', i)
        vic_acc.put({
            "image" : np.moveaxis(np.array(zf["data/image"][i]),-1,0),  # swap axis to make it fit the dataset shape
            "robot_obs" : np.array(zf["data/robot_obs"][i])
        })

        # print(vic_acc.get())
        np_obs_dict = dict(vic_acc.get())
        # print(np_obs_dict)
        # np_obs_dict = dict(obs) 
        # for k, v in np_obs_dict.items():
        #     print(k, v.shape)
        # print(np_obs_dict["image"].shape)


        # device transfer
        obs_dict = dict_apply(np_obs_dict,  # type: ignore
            lambda x: torch.from_numpy(x).to(
                device=device))

        t1 = time.time()
        # run policy
        with torch.no_grad():
            action_dict = policy.predict_action(obs_dict)
        t2 = time.time()
        print("inference time:", t2 - t1, "seconds")

        # device_transfer
        np_action_dict = dict_apply(action_dict,
            lambda x: x.detach().to('cpu').numpy())

        action = np_action_dict['action']
        print(action.shape)
        print("pred action:", action[0][0])
        print("true action:", zf["data/robot_act"][i])
        print("delta:", action[0][0] - zf["data/robot_act"][i])
        print("percentage off:", np.abs((action[0][0] - zf["data/robot_act"][i])) / zf["data/robot_act"][i])
        print("absolute delta sum:", np.sum(np.abs((action[0][0] - zf["data/robot_act"][i]))))