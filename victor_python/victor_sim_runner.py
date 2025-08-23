# this sim test takes in the observation input from the sim 
# and predicts future actions based on it

from abc import ABC, abstractmethod
import threading
from enum import IntEnum
import time
from time import perf_counter, sleep
from queue import Queue, Empty
from typing import Optional
import argparse

import dill
import hydra
import numpy as np
import torch
from tqdm import tqdm

import rclpy
from sensor_msgs.msg import Image
from geometry_msgs.msg import WrenchStamped
import ros2_numpy as rnp

from diffusion_policy.common.victor_accumulator import ObsAccumulator
import zarr
from victor_python.victor_policy_client import VictorPolicyClient
from diffusion_policy.common.pytorch_util import dict_apply
from data_utils import SmartDict, store_h5_dict

from diffusion_policy.workspace.base_workspace import BaseWorkspace

class ObsManager(ABC):
    obs_type="base"
    def __init__(self,
        client,
        side,
        device,
        n_obs_steps,
        n_action_steps,
    ):
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps

        self.client = client
        self.side = side
        self.arm = self.client.right if self.side == 'right' else self.client.left
        self.device = device

        self.data_dict = SmartDict()
        self.accumulator = ObsAccumulator(n_obs_steps)

        self.latest_img = None

    def get_live_obs(self):
        assert self.client.__getattribute__(self.side) is not None, self.side + " arm client is not initialized"
        assert self.arm is not None
        
        joint_status = self.arm.get_joint_positions()
        gripper_obs = self.arm.get_gripper_positions()

        if gripper_obs is None or joint_status is None: 
            return None, None

        # Build action tensor
        joint_cmd = self.arm.get_joint_cmd()
        gripper_cmd = self.arm.get_gripper_cmd()
        curr_act = np.concatenate([
            joint_cmd if joint_cmd is not None else joint_status,
            gripper_cmd[0:1] if gripper_cmd is not None else gripper_obs[0:1]
        ])
        self.previous_act = curr_act[:8]

        # build observation tensor
        sim_obs = np.concatenate([curr_act, joint_status, gripper_obs])
        return sim_obs, curr_act

    @abstractmethod
    def get_obs(self) -> dict:
        return {}

    def add_actions(self, action_tensor):
        for j in range(self.n_action_steps):
            if np.any(np.isnan(action_tensor[0][j])):
                continue # if there is a nan, skip command
            self.data_dict.add('robot_act', action_tensor[0][j])  # store the action in the data_dict

    def destroy(self):
        # Clean up subscribers and any other resources
        # Implementation for saving live data goes here
        timestamp = time.strftime("%m-%d %H:%M:%S", time.localtime())
        store_h5_dict(f"data/victor_eval_output/{self.obs_type}_run_{timestamp}.h5", self.data_dict)

class PlaybackObsManager(ObsManager):
    obs_type="replay"
    def __init__(self,
        client,
        side,
        device,
        n_obs_steps,
        n_action_steps,
        zf,
        traj_idx,
    ):
        super().__init__(client, side, device, n_obs_steps, n_action_steps)
        self.zf = zf
        self.traj_idx = traj_idx

        # Find the start and end based on index
        episode_ends = self.zf["meta/episode_ends"]
        if self.traj_idx < 0 or self.traj_idx >= len(episode_ends):
            raise ValueError(f"traj_idx {self.traj_idx} is out of bounds for dataset with {len(episode_ends)} episodes")
        
        self.start_idx = 0 if self.traj_idx == 0 else episode_ends[self.traj_idx - 1]
        self.end_idx = episode_ends[self.traj_idx]
        self.time_idx = self.start_idx

    def get_obs(self) -> dict:
        if self.time_idx >= self.end_idx:
            return {}
        
        sim_obs, _ = super().get_live_obs()
        if sim_obs is None:
            return {}
        
        self.data_dict.add('robot_obs', sim_obs)
        self.accumulator.put({
            "image" : np.moveaxis(np.array(self.zf["data/image"][self.time_idx]),-1,0)/255,  # swap axis to make it fit the dataset shape
            "robot_obs" : np.array(self.zf["data/robot_obs"][self.time_idx])
        })
        self.time_idx += 1

        # get and device transfer
        np_obs_dict = self.accumulator.get()
        obs_dict = dict_apply(np_obs_dict,  # type: ignore
            lambda x: torch.from_numpy(x).to(
                device=self.device))
        return obs_dict
    
    def get_playback_length(self):
        return self.end_idx - self.start_idx
    
class LiveObsManager(ObsManager):
    obs_type="live"
    def __init__(self, 
        client: VictorPolicyClient,
        side: str,
        device: torch.device,
        n_obs_steps: int,
        n_action_steps: int,
    ):
        super().__init__(client, side, device, n_obs_steps, n_action_steps)
        self.latest_img = None
        self._setup_subscribers()

    def get_obs(self):
        # Implement the logic to get real-time observations from the robot arm
        np_obs_dict = self.accumulator.get()
        
        # Check if accumulator has enough observations
        if not np_obs_dict or len(np_obs_dict) == 0:
            return {}
        
        # Verify all required keys are present and have data
        for key in ['image', 'robot_obs']:
            if key not in np_obs_dict or np_obs_dict[key] is None:
                return {}
        
        obs_dict = dict_apply(np_obs_dict,  # type: ignore
            lambda x: torch.from_numpy(x).to(
                device=self.device))
        return obs_dict

    # sets up subscribers that are needed for the model specifically
    def _setup_subscribers(self):
        self.client.img_sub = self.client.create_subscription(
            Image,
            '/zivid/rgb',
            self.image_callback,
            self.client.reliable_qos
        )
        self.client.wrench_stop_sub = self.client.create_subscription(
            WrenchStamped,
            f'/victor/{self.side}_arm/wrench',
            self.wrench_stop_callback,
            self.client.reliable_qos
        )

    def wrench_stop_callback(self, msg: WrenchStamped):
        w = msg.wrench
        wrench = [w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z]
        if np.any(np.abs(wrench[:3]) > 55):
            exit(1) # exit the execution 
    
    def image_callback(self, msg: Image):
        received_t = perf_counter()
        self.latest_img = rnp.numpify(msg)
        self.last_img_t = received_t
        self.update_accumulator_callback()

    def update_accumulator_callback(self):
        sim_obs, _ = super().get_live_obs()
        if sim_obs is None:
            return

        assert self.latest_img is not None, "Latest image is None, did you receive an image?"
        self.data_dict.add("image", self.latest_img)
        self.accumulator.put({
            "image" : np.moveaxis(self.latest_img,-1,0)/255,  # swap axis to make it fit the dataset shape
            "robot_obs" : sim_obs
        })
        self.data_dict.add('robot_obs', sim_obs)

class ActionManager:
    def __init__(self, arm, action_shape: tuple, n_action_steps, hz=10):
        self.arm = arm
        self.action_shape = action_shape
        self.n_action_steps = n_action_steps
        self.frequency = 1.0 / hz

        self.action_lock = threading.Lock()
        self.actions = Queue()

        # Start a thread to send actions in the background
        self._running = True
        self._thread = threading.Thread(target=self._execute, daemon=True)
        self._thread.start()

    def add_actions(self, action_tensor):
        actions = []
        # Remove print statement to reduce CPU usage
        # print("Action shape", action_tensor.shape, self.n_action_steps)
        for j in range(self.n_action_steps):
            if np.any(np.isnan(action_tensor[0][j])):
                continue # if there is a nan, skip command
            actions.append(action_tensor[0][j])

        with self.action_lock:
            while not self.actions.empty():
                self.actions.get()  # Clear the queue if it is not empty
            for action in actions:
                self.actions.put(action)
    
    def _execute(self):
        while self._running:
            try:
                # Use blocking get with timeout to reduce CPU usage
                action = self.actions.get(timeout=0.5)
                self._send_action(action)
                sleep(self.frequency)
            except Empty:
                # No actions to process, sleep longer to reduce CPU usage
                sleep(0.1)
            except Exception as e:
                print(f"Error executing action: {e}")
                sleep(0.1)

    # assumes [7 dim joint angles, 1 dim gripper]
    def _send_action(self, action):
        assert self.arm is not None
        assert action.shape == (8,)
        self.arm.set_joint_cmd(np.array(action[:7]))
        self.arm.set_gripper_cmd(np.array([action[7], action[7], action[7], 1]))

    def stop(self):
        self._running = False
        self._thread.join()


class RunningMode(IntEnum):
    REPLAY = 0
    LIVE = 1

class VictorSimClient:
    def __init__(self, 
        running_mode: RunningMode=RunningMode.LIVE, 
        side: str='right', 
        device: str='cuda',
        replay_dataset: Optional[str] = 'data/victor/victor_data_08_06_new50_no_interp.zarr.zip',
        replay_idx: Optional[int] = None
    ):
        # Initialize client with both arms enabled and specified device
        assert running_mode in (RunningMode.REPLAY, RunningMode.LIVE), \
            f"Invalid running_mode: {running_mode}"
        self.running_mode = running_mode
        self.side = side
        self.device = torch.device(device)

        ### SETUP POLICY
        output_dir = "data/victor_eval_output"
        import pathlib
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # LOAD CHECKPOINT
        payload = torch.load(open("data/outputs/2025.08.06/17.50.31_victor_diffusion_image_victor_diff/checkpoints/latest.ckpt", "rb"), pickle_module=dill)

        self.cfg = payload['cfg']
        self.cfg.policy.n_action_steps = 1          # Overwite due to training error
        self.cfg.policy.num_inference_steps = 10
        cls = hydra.utils.get_class(self.cfg._target_)
        workspace = cls(self.cfg, output_dir=output_dir)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # get policy from workspace
        self.policy = workspace.model # type: ignore
        if self.cfg.training.use_ema:
            self.policy = workspace.ema_model # type: ignore
        # self.device = self.policy.device
        print("Running on device", self.device)
        self.policy.to(self.device)

        # Initialize ROS stuff
        self.client = VictorPolicyClient('victor_inference', enable_left = self.side == 'left',
                                            enable_right = self.side == 'right', device=device)
        self.arm = self.client.right if self.side == 'right' else self.client.left
        self.get_logger = self.client.get_logger

        # Setup observation and action managers
        if self.running_mode == RunningMode.REPLAY:
            if replay_dataset is None:
                raise ValueError("replay_dataset must be provided in REPLAY mode")
            if replay_idx is None:
                raise ValueError("replay_idx must be provided in REPLAY mode")
            zf = zarr.open(replay_dataset, mode='r')
            self.obs_manager = PlaybackObsManager(
                self.client, self.side, self.device,
                n_obs_steps=self.cfg.policy.n_obs_steps, 
                n_action_steps=self.cfg.policy.n_action_steps,
                zf=zf,
                traj_idx=replay_idx
            )
            print("Replaying trajectory", zf["meta/episode_name"])
        elif self.running_mode == RunningMode.LIVE:
            self.obs_manager = LiveObsManager(
                self.client, self.side, self.device,
                n_obs_steps=self.cfg.policy.n_obs_steps, 
                n_action_steps=self.cfg.policy.n_action_steps,
            )
        else:
            raise ValueError(f"Unsupported running mode: {running_mode}")
        
        # Setup action manager
        self.action_manager = ActionManager(
            self.arm, action_shape=(8,), n_action_steps=self.cfg.policy.n_action_steps, hz=10
        )

        # Setup looping params
        if self.running_mode == RunningMode.LIVE:
            self.max_iteration = 6000       # max 10 min at 10 Hz
        elif self.running_mode == RunningMode.REPLAY:
            assert isinstance(self.obs_manager, PlaybackObsManager), \
                "obs_manager must be an instance of PlaybackObsManager in REPLAY mode"
            self.max_iteration = self.obs_manager.get_playback_length()
        else:
            raise ValueError(f"Unsupported running mode: {self.running_mode}")

    def wait_for_server(self, timeout: float = 10.0) -> bool:
        """Wait for server to be available and sending status."""
        # Check if server topics exist
        if not self.client.check_server_availability():
            self.get_logger().error("Server topics not found - is VictorPolicyServer running?")
            return False
        # Wait for status messages
        if not self.client.wait_for_status(timeout):
            self.get_logger().warn("No status received from server, but continuing anyway...")
            return True  # Continue even if no status - server might be in dry run mode
        # Check what status we received
        self.get_logger().info("Server is responding!")
        status = self.client.get_combined_status()
        if status:
            self.get_logger().info(f"Server status: {status}")
        return True
        
    def run_trajectory_inference_example(self):
        """Example of controlling arms individually with different controllers using new API."""
        # Check server status
        if not self.wait_for_server(10.0):
            return
        
        # Check arm
        if not self.arm:
            self.get_logger().error(f"{self.side} arm is not initialized")
            return
        
        if not self.client.set_controller(self.side, 'impedance_controller', timeout=10.0):
            self.get_logger().error(f"Failed to switch {self.side} arm controller")
            return

                # Control loop
        self.cfg.n_action_steps = 1
        pbar = tqdm(range(self.max_iteration))
        target_hz = 10  # Control loop frequency
        target_dt = 1.0 / target_hz
        
        for i in pbar:
            loop_start = perf_counter()
            
            # TODO pause the observation timer here? so that way the robot doesnt have huge gaps between but probably a bad idea
            obs_dict = self.obs_manager.get_obs()
            
            # Skip if observations aren't ready yet
            if not obs_dict or len(obs_dict) == 0:
                pbar.set_description("Waiting for observations...")
                sleep(0.1)  # Small delay to avoid busy waiting
                continue

            st = perf_counter()
            with torch.no_grad():
                action_dict = self.policy.predict_action(obs_dict)
            # Update tqdm with inference time
            inference_time = perf_counter() - st
            pbar.set_description(f"T: {inference_time:.3f} s")

            # device_transfer
            np_action_dict = dict_apply(action_dict,
                lambda x: x.detach().to('cpu').numpy())
            action = np_action_dict['action']

            self.obs_manager.add_actions(action)
            self.action_manager.add_actions(action)
            
            # Rate limiting to target frequency
            loop_time = perf_counter() - loop_start
            remaining_time = target_dt - loop_time
            if remaining_time > 0:
                sleep(remaining_time)

    def destroy(self):
        self.client.destroy_node()
        self.obs_manager.destroy()

# iter: 0
# OBS:
#  [ 1.70956299  0.50377883 -1.66476116 -1.02923538 -0.41833602  1.45144483
#   1.21576354  0.36862745  0.36862745  0.36862745  1.          1.71155818
#   0.50316002 -1.6632373  -1.02566312 -0.41926303  1.44945563  1.21469737
#   0.36862745  0.36862745  0.36862745  0.83137255]
# inference time: 0.12793731689453125 seconds
# (1, 1, 11)
# action: [ 1.6682425   0.45750433 -1.674718   -0.9863463  -0.41966382  1.49741
#   1.1789628   0.36862746  0.36862746  0.36862746  1.0039515 ]
# true action: [ 1.70956299  0.50377883 -1.66476116 -1.02923538 -0.41833602  1.45144483
#   1.21576354  0.36862745  0.36862745  0.36862745  1.        ]

def main(args=None):
    # Parse CLI args for running mode and replay dataset. Unknown args will be forwarded to rclpy.
    parser = argparse.ArgumentParser(description="Victor sim runner")
    parser.add_argument(
        "--mode", choices=["live", "replay"], default="live",
        help="Running mode: 'live' for real robot, 'replay' for dataset playback"
    )
    parser.add_argument(
        "--replay_dataset", type=str,
        default="data/victor/victor_data_08_06_new50_no_interp.zarr.zip",
        help="Path to replay dataset (used when --mode replay)"
    )
    parser.add_argument(
        "--replay_idx", type=int, default=0,
        help="Trajectory index to replay from the dataset (used when --mode replay). Defaults to 0."
    )

    parsed_args, remaining = parser.parse_known_args(args)

    # Initialize ROS with remaining args
    rclpy.init(args=remaining)

    victor_sim = None
    try:
        enable_left, enable_right = False, True

        # Convert mode to enum and pass replay_dataset when appropriate
        running_mode = RunningMode.LIVE if parsed_args.mode == "live" else RunningMode.REPLAY
        replay_ds = parsed_args.replay_dataset if running_mode == RunningMode.REPLAY else None
        replay_idx = parsed_args.replay_idx if running_mode == RunningMode.REPLAY else None

        # Initialize example with specified configuration
        victor_sim = VictorSimClient(
            running_mode=running_mode,
            device='cuda:0',
            replay_dataset=replay_ds,
            replay_idx=replay_idx
        )
        victor_sim.get_logger = victor_sim.client.get_logger
        
        # Start ROS spinning in a separate thread
        import threading
        from rclpy.executors import MultiThreadedExecutor
        
        executor = MultiThreadedExecutor()
        executor.add_node(victor_sim.client)
        
        # Start spinning in a separate thread
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()
        
        victor_sim.get_logger().info(f"Running examples on device: {victor_sim.device}")
        victor_sim.get_logger().info(f"Arms enabled - Left: {enable_left}, Right: {enable_right}")
        
        victor_sim.run_trajectory_inference_example()
        victor_sim.destroy()

    except KeyboardInterrupt:
        if victor_sim:
            victor_sim.get_logger().info("Interrupted by user")
            victor_sim.destroy()
    except Exception as e:
        import traceback
        full_traceback = traceback.format_exc()
        print(f"Client error: {e}")
        print(f"Full traceback:\n{full_traceback}")
        if victor_sim:
            victor_sim.get_logger().error(f"Exception: {e}")
            victor_sim.get_logger().error(f"Full traceback:\n{full_traceback}")
    finally:
        if victor_sim is not None:
            victor_sim.client.destroy_node()
        rclpy.shutdown()        

if __name__ == "__main__":
    main()