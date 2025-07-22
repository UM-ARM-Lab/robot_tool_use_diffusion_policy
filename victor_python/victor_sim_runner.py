# this sim test takes in the observation input from the sim 
# and predicts future actions based on it

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from threading import Lock
import json
import time
import argparse
import dill
import hydra
from typing import Dict, Any, Callable, Optional, Union, List
import numpy as np
import torch
import pathlib
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import TransformStamped, WrenchStamped
from victor_hardware_interfaces.msg import (
    MotionStatus, 
    Robotiq3FingerCommand, 
    Robotiq3FingerStatus,
    JointValueQuantity,
    Robotiq3FingerActuatorCommand
)

import ros2_numpy as rnp

from victor_python.victor_utils import wrench_to_tensor, gripper_status_to_tensor

from diffusion_policy.common.victor_accumulator import ObsAccumulator
import zarr
from victor_python.victor_policy_client import VictorPolicyClient
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.workspace.base_workspace import BaseWorkspace

class VictorSimClient:
    def __init__(self, device: Union[str, torch.device] = 'cpu'):
        # Initialize client with both arms enabled and specified device
        self.side = 'left'
        self.client = VictorPolicyClient('policy_example', enable_left = self.side == 'left',
                                            enable_right = self.side == 'right', device=device)
        # self.arm_node = self.client.left.node if self.side == 'left' else self.client.right.node # type: ignore
        self.get_logger = self.client.get_logger
        # self.device = torch.device(device)    # use model device
        self.accumulator = ObsAccumulator(2)

        ### SETUP POLICY
        output_dir = "data/victor_eval_output"
        # if os.path.exists(output_dir):
        #     click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        # load checkpoint
        # payload = torch.load(open("data/outputs/2025.06.19/12.52.40_train_diffusion_unet_hybrid_victor_diff/checkpoints/epoch=0250-train_action_mse_error=0.000.ckpt", 'rb'), pickle_module=dill)
        # payload = torch.load(open("data/outputs/2025.06.23/15.33.39_train_diffusion_unet_hybrid_victor_diff/checkpoints/latest.ckpt", 'rb'), pickle_module=dill)
        # payload = torch.load(open("data/outputs/2025.06.24/12.24.58_train_diffusion_unet_hybrid_victor_diff/checkpoints/latest.ckpt", 'rb'), pickle_module=dill)
        # payload = torch.load(open("data/outputs/2025.06.24/12.51.10_train_diffusion_unet_hybrid_victor_diff/checkpoints/latest.ckpt", 'rb'), pickle_module=dill)
        # DDIM
        
        # payload = torch.load(open("data/outputs/2025.06.24/13.31.55_train_diffusion_unet_hybrid_victor_diff/checkpoints/latest.ckpt", 'rb'), pickle_module=dill)
        # lower down_dims
        # payload = torch.load(open("data/outputs/2025.06.24/13.58.39_train_diffusion_unet_hybrid_victor_diff/checkpoints/latest.ckpt", 'rb'), pickle_module=dill)
        # lower down_dims and lower action horizon (since we only use 2 atm) and sample prediction type
        payload = torch.load(open("data/outputs/2025.06.24/14.21.53_train_diffusion_unet_hybrid_victor_diff/checkpoints/latest.ckpt", 'rb'), pickle_module=dill)
        # high down_dims and sample prediction BAD
        # payload = torch.load(open("data/outputs/2025.06.25/12.27.36_train_diffusion_unet_hybrid_victor_diff/checkpoints/latest.ckpt", 'rb'), pickle_module=dill)
        
        cfg = payload['cfg']
        cfg.policy.num_inference_steps = 20
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=output_dir)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        
        # get policy from workspace
        self.policy = workspace.model # type: ignore
        if cfg.training.use_ema:
            self.policy = workspace.ema_model # type: ignore
        self.device = self.policy.device

        self.device = torch.device(self.device)
        self.policy.to(self.device)
        # policy.eval()
        # self.zf = zarr.open("data/victor/victor_data.zarr", mode='r') #"data/pusht/pusht_cchi_v7_replay.zarr"
        # self.zf = zarr.open("data/victor/victor_state_data.zarr", mode='r') #"data/pusht/pusht_cchi_v7_replay.zarr"
        self.zf = zarr.open("data/victor/victor_state_data_0624.zarr", mode='r') #"data/pusht/pusht_cchi_v7_replay.zarr"

    # sets up subscribers that are needed for the model specifically
    def _setup_subscribers(self):
        self.img_sub = self.client.create_subscription(
            Image,
            '/fake_image',
            self.image_callback,
            self.client.reliable_qos
        )
    
    def image_callback(self, msg: Image):
        self.latest_img = rnp.numpify(msg)
        return 

    def wait_for_server(self, timeout: float = 10.0) -> bool:
        """Wait for server to be available and sending status."""
        self.get_logger().info("Checking for server availability...")
        
        # Check if server topics exist
        if not self.client.check_server_availability():
            self.get_logger().error("Server topics not found - is VictorPolicyServer running?")
            return False
        
        self.get_logger().info("Server topics found, waiting for status messages...")
        
        # Wait for status messages
        if self.client.wait_for_status(timeout):
            self.get_logger().info("Server is responding!")
            
            # Check what status we received
            status = self.client.get_combined_status()
            if status:
                self.get_logger().info(f"Server status: {status}")
            
            return True
        else:
            self.get_logger().warn("No status received from server, but continuing anyway...")
            return True  # Continue even if no status - server might be in dry run mode
    
    def run_trajectory_inference_example(self):
        """Example of controlling arms individually with different controllers using new API."""
        self.get_logger().info("Starting individual arm control example...")
        
        if not self.wait_for_server(10.0):
            self.get_logger().error("Failed to connect to server")
            return
        
        # Individual arm controller switching using new centralized API
        if self.client.left:
            if not self.client.set_controller('left', 'impedance_controller', timeout=10.0):
                self.get_logger().error("Failed to switch left arm controller")
                return
        
        # Control loop
        for i in range(696):  
            print('iter:', i)
            left_pos = self.client.left.get_joint_positions() # type: ignore
            # print(left_pos)
            left_wrench = self.client.left.get_wrench() # type: ignore
            wrench = wrench_to_tensor(left_wrench, self.client.device) # type: ignore
            # print(wrench)
            left_gripper = self.client.left.get_gripper_status() # type: ignore
            gripper_obs = gripper_status_to_tensor(left_gripper, self.client.device) # type: ignore
            # print(gripper_obs)
            sim_obs = np.hstack([left_pos, gripper_obs, wrench])
            print("SIM OBS:\n", sim_obs)
            if left_pos is not None:
                self.accumulator.put({
                    "image" : np.moveaxis(np.array(self.zf["data/image"][i]),-1,0),  # swap axis to make it fit the dataset shape
                    # "robot_obs" : np.array(self.zf["data/robot_obs"][i])
                    "robot_obs" : sim_obs
                })
                print("ROBOT_OBS:\n", np.array(self.zf["data/robot_obs"][i]))

                np_obs_dict = dict(self.accumulator.get())

                # device transfer
                obs_dict = dict_apply(np_obs_dict,  # type: ignore
                    lambda x: torch.from_numpy(x).to(
                        device=self.device))
                

                t1 = time.time()
                # run policy
                with torch.no_grad():
                    action_dict = self.policy.predict_action(obs_dict)
                t2 = time.time()
                print("inference time:", t2 - t1, "seconds")

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())
                
                action = np_action_dict['action']
                self.client.left.send_joint_command(action[0][0][:7])
                self.client.left.send_gripper_command(action[0][0][7:])

                print("pred action:", action[0][0])
                print("true action:", self.zf["data/robot_act"][i])

        self.get_logger().info("Individual arm control example completed")

        
def main(args=None):
    # Initialize ROS with remaining args
    rclpy.init(args=args)

    victor_sim = None
    try:
        enable_left, enable_right = True, False

        # Initialize example with specified configuration
        victor_sim = VictorSimClient()
        victor_sim.client = VictorPolicyClient(
            'policy_example', 
            enable_left=enable_left, 
            enable_right=enable_right, 
            device=victor_sim.device
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
        
    except KeyboardInterrupt:
        if victor_sim:
            victor_sim.get_logger().info("Interrupted by user")
    except Exception as e:
        print(f"Client error: {e}")
        if victor_sim:
            victor_sim.get_logger().error(f"Exception: {e}")
    finally:
        if victor_sim is not None:
            victor_sim.client.destroy_node()
        rclpy.shutdown()        

if __name__ == "__main__":
    main()