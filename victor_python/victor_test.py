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
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
from data_utils import SmartDict, store_h5_dict, store_zarr_dict


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)

    output_dir = "data/victor_eval_output"
    # if os.path.exists(output_dir):
    #     click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    data_dict = SmartDict()

    ### 15 EPISODES no wrench
    # 30 epoch image + state
    # payload = torch.load(open("data/outputs/2025.07.18/13.44.32_victor_diffusion_image_victor_diff/checkpoints/latest.ckpt", "rb"), pickle_module=dill)
    # 250 epoch image + state OVERFIT
    # payload = torch.load(open("data/outputs/2025.07.20/12.01.12_victor_diffusion_image_victor_diff/checkpoints/latest.ckpt", "rb"), pickle_module=dill)
    # 30 epoch image + state LOW VAL LOSS
    # payload = torch.load(open("data/outputs/2025.07.20/12.01.12_victor_diffusion_image_victor_diff/checkpoints/epoch=0030-train_action_mse_error=0.000.ckpt", "rb"), pickle_module=dill)

    
    # 30 epoch state only
    # payload = torch.load(open("data/outputs/2025.07.20/11.03.03_victor_diffusion_state_victor_diff/checkpoints/latest.ckpt", "rb"), pickle_module=dill)
    # 60 epoch state only
    # payload = torch.load(open("data/outputs/2025.07.21/09.57.34_victor_diffusion_state_victor_diff/checkpoints/latest.ckpt", "rb"), pickle_module=dill)


    # NEW OBS CONFIG TODO
    # 50 epoch state only
    # payload = torch.load(open("data/outputs/2025.07.22/13.15.22_victor_diffusion_state_victor_diff/checkpoints/latest.ckpt", "rb"), pickle_module=dill)
    # 2620 epoch state only
    # payload = torch.load(open("data/outputs/2025.07.24/16.59.38_victor_diffusion_state_victor_diff/checkpoints/epoch=2620-train_action_mse_error=0.0000040.ckpt", "rb"), pickle_module=dill)
    # 555 epoch state only -> SINGLE OBS STEP
    # payload = torch.load(open("data/outputs/2025.07.25/16.14.19_victor_diffusion_state_micro_victor_diff/checkpoints/latest.ckpt", "rb"), pickle_module=dill)
        
    # NEW NEW EA OBS CONFIG
    # payload = torch.load(open("data/outputs/2025.07.28/16.39.25_victor_diffusion_state_victor_diff/checkpoints/latest.ckpt", "rb"), pickle_module=dill)
    
    # OLD NEW OBS CONFIG (Joint angles)
    # 500 epochs image + sample
    # payload = torch.load(open("data/outputs/2025.07.29/16.18.40_victor_diffusion_image_victor_diff/checkpoints/latest.ckpt", "rb"), pickle_module=dill)
    # 215 epochs image + sample + NO PLATEAUS
    payload = torch.load(open("data/outputs/2025.07.31/19.19.04_victor_diffusion_image_victor_diff/checkpoints/latest.ckpt", "rb"), pickle_module=dill)

    cfg = payload['cfg']
    cfg.policy.num_inference_steps = 16
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

    ### DATASET
    # zf = zarr.open("data/victor/victor_data_07_24_single_trajectory.zarr", mode='r') 
    # zf = zarr.open("data/victor/victor_data_07_28_end_affector.zarr", mode='r') 
    # zf = zarr.open("data/victor/victor_data_07_29_all_ep_ea.zarr", mode='r') 
    zf = zarr.open("data/victor/victor_data_07_31_no_plat.zarr", mode='r') 

    vic_acc = ObsAccumulator(cfg.policy.n_obs_steps)

    for i in range(2225, 2235):    #10535, 11193
        print('iter:', i)

        vic_acc.put({
            "image" : np.moveaxis(np.array(zf["data/image"][i]),-1,0)/255,  # swap axis to make it fit the dataset shape
            "robot_obs" : np.array(zf["data/robot_obs"][i])
        })

        data_dict.add("data/image", np.moveaxis(np.array(zf["data/image"][i]),-1,0)/255)
        data_dict.add("data/robot_obs", np.array(zf["data/robot_obs"][i]))

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

        action = np_action_dict['action_pred'][0][0]
        # NOTE: in real life, the robot executes n_action_steps at a time instead of only 1. However, when
        #       testing it without live feedback it doesn't make sense to do so since it can't get feedback on its actions
        data_dict.add("data/robot_act_pred", action)
        # print(np_action_dict["action"], np_action_dict["action_pred"])
        # print(action[:7], "\t", action[7:])  # first 7 are joint positions, last 3 are gripper positions
        print("pred action:", action)
        print("true action:", zf["data/robot_act"][i])
        print("delta:", action - zf["data/robot_act"][i])
        print("percentage off:", np.abs((action - zf["data/robot_act"][i])) / zf["data/robot_act"][i])
        print("absolute delta sum:", np.sum((action - zf["data/robot_act"][i])**2))
    
    store_h5_dict("data/victor/victor_test.h5", data_dict)