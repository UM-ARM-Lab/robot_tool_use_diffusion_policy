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
import cv2
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
        self.side = 'right'
        self.client = VictorPolicyClient('policy_example', enable_left = self.side == 'left',
                                            enable_right = self.side == 'right', device=device)
        self._setup_subscribers()
        self.latest_img = None
        # self.arm_node = self.client.left.node if self.side == 'left' else self.client.right.node # type: ignore
        self.get_logger = self.client.get_logger
        # self.device = torch.device(device)    # use model device
        self.accumulator = ObsAccumulator(1)

        ### SETUP POLICY
        output_dir = "data/victor_eval_output"
        # if os.path.exists(output_dir):
        #     click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        ### 15 EPISODES no wrench
        # 30 epoch image + state
        # payload = torch.load(open("datImagea/outputs/2025.07.18/13.44.3episode_ends2_victor_diffusion_image_victor_diff/checkpoints/latest.ckpt", "rb"), pickle_module=dill)
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
        # 210 epoch image + state
        # payload = torch.load(open("data/outputs/2025.07.22/17.14.10_victor_diffusion_image_victor_diff/checkpoints/epoch=0180-train_action_mse_error=0.0000083.ckpt", "rb"), pickle_module=dill)
        # 100 epoch image + state + epsilon
        # payload = torch.load(open("data/outputs/2025.07.24/15.45.20_victor_diffusion_state_victor_diff/checkpoints/latest.ckpt", "rb"), pickle_module=dill)
        # 2620 epoch state only
        # payload = torch.load(open("data/outputs/2025.07.24/16.59.38_victor_diffusion_state_victor_diff/checkpoints/epoch=2620-train_action_mse_error=0.0000040.ckpt", "rb"), pickle_module=dill)
        # 555 epoch state only -> SINGLE OBS STEP
        payload = torch.load(open("data/outputs/2025.07.25/12.16.22_victor_diffusion_state_victor_diff/checkpoints/epoch=0555-train_action_mse_error=0.0001751.ckpt", "rb"), pickle_module=dill)

        cfg = payload['cfg']
        cfg.policy.num_inference_steps = 100
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

        self.zf = zarr.open("data/victor/victor_data_07_24_single_trajectory.zarr", mode='r') 
        # self.zf = zarr.open("data/victor/victor_data_07_22_no_wrench.zarr", mode='r') 

    # sets up subscribers that are needed for the model specifically
    def _setup_subscribers(self):
        self.client.img_sub = self.client.create_subscription(
            Image,
            '/zivid/rgb',
            self.image_callback,
            self.client.reliable_qos
        )
    
    def image_callback(self, msg: Image):
        # print("Image received")
        # if self.latest_img is not None:
        #     print("Image already received, skipping")
        #     return
        self.latest_img = rnp.numpify(msg)
        # print(self.latest_img.shape)
        # print(self.latest_img)
        # cv2.namedWindow("Latest Image", cv2.WINDOW_NORMAL)
        # self.latest_img = self.latest_img[...,::-1]
        # cv2.imshow("Latest Image", cv2.cvtColor(self.latest_img, cv2.COLOR_BGR2RGB))
        # cv2.imshow("Latest Image", self.latest_img)
        # cv2.waitKey(1)  # Add small delay to allow window to update
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
        # if self.client.left:
        #     if not self.client.set_controller('left', 'impedance_controller', timeout=10.0):
        #         self.get_logger().error("Failed to switch left arm controller")
        #         return
        print("s", self.client.right)
        if self.client.right:
            if not self.client.set_controller('right', 'impedance_controller', timeout=10.0):
                self.get_logger().error("Failed to switch left arm controller")
                return
        
        # previous_act = np.array(self.zf["data/robot_act"][0])
        print("OLD previous_act:", self.zf["data/robot_act"][0])
        action = self.zf["data/robot_act"][0]
        previous_act = action
        self.client.right.send_joint_command(action[:7])
        self.client.right.send_gripper_command(action[7:])
        time.sleep(1)  # wait for the action to be sent
        # previous_act = None
        # Control loop
        for i in range(789000):  
            print('iter:', i)

            right_pos = self.client.right.get_joint_positions() # type: ignore
            rms = self.client.right.get_motion_status().commanded_joint_position
            right_motion_status = np.array([
                rms.joint_1, rms.joint_2, rms.joint_3, 
                rms.joint_4, rms.joint_5, rms.joint_6, 
                rms.joint_7
            ])

            # print(self.client.right.get_motion_status().commanded_joint_position.get_fields_and_field_types())
            right_gripper = self.client.right.get_gripper_status() # type: ignore
            gripper_obs = gripper_status_to_tensor(right_gripper, self.client.device) # type: ignore
            
            if previous_act is None:
                previous_act = np.hstack([right_pos, gripper_obs[0], gripper_obs[1], gripper_obs[2], gripper_obs[3]])
                # previous_act = np.hstack([right_pos, 0, 0, 0, 0])
                print("NEW previous_act:", previous_act)
                continue
            # print(wrench)
            # right_pos = self.client.right.get_joint_positions() # type: ignore
            # right_gripper = self.client.right.get_gripper_status() # type: ignore
            gripper_obs = gripper_status_to_tensor(right_gripper, self.client.device) # type: ignore
            sim_obs = np.hstack([previous_act, right_motion_status, gripper_obs[1], gripper_obs[3], gripper_obs[5], gripper_obs[7]])
            print(sim_obs.shape)
            print("SIM OBS:\n", sim_obs)
            # time.sleep(1)  # simulate some delay
            # continue
            if right_pos is not None:
                self.accumulator.put({
                    "image" : np.moveaxis(np.array(self.zf["data/image"][i]),-1,0)/255,  # swap axis to make it fit the dataset shape
                    # "image": np.moveaxis(self.latest_img,-1,0),
                    # "robot_obs" : np.array(self.zf["data/robot_obs"][i])
                    # "image" : np.moveaxis(self.latest_img,-1,0)/255,  # swap axis to make it fit the dataset shape
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
                self.client.right.send_joint_command(action[0][0][:7])
                self.client.right.send_gripper_command(action[0][0][7:])
                previous_act = action[0][0]

                print("pred action:", action[0][0],"\n")
                print("true action:", self.zf["data/robot_act"][i])

        self.get_logger().info("Individual arm control example completed")

        
def main(args=None):
    # Initialize ROS with remaining args
    rclpy.init(args=args)

    victor_sim = None
    try:
        enable_left, enable_right = False, True

        # Initialize example with specified configuration
        victor_sim = VictorSimClient()
        # victor_sim.client = VictorPolicyClient(
        #     'policy_example', 
        #     enable_left=enable_left, 
        #     enable_right=enable_right, 
        #     device=victor_sim.device
        # )
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