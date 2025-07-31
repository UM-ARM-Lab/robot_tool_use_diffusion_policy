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
from data_utils import SmartDict, store_h5_dict

from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.workspace.base_workspace import BaseWorkspace

class VictorSimClient:
    def __init__(self, device: Union[str, torch.device] = 'cpu'):
        # Initialize client with both arms enabled and specified device
        self.side = 'right'
        self.client = VictorPolicyClient('policy_example', enable_left = self.side == 'left',
                                            enable_right = self.side == 'right', device=device)
        self.arm = self.client.right if self.side == 'right' else self.client.left

        self._setup_subscribers()
        self.latest_img = None
        # self.arm_node = self.client.left.node if self.side == 'left' else self.client.right.node # type: ignore
        self.get_logger = self.client.get_logger
        self.data_dict = SmartDict()
        # self.device = torch.device(device)    # use model device

        ### SETUP POLICY
        output_dir = "data/victor_eval_output"
        # if os.path.exists(output_dir):
        #     click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # LOAD CHECKPOINT
        # 2620 epoch state only
        # payload = torch.load(open("data/outputs/2025.07.24/16.59.38_victor_diffusion_state_victor_diff/checkpoints/epoch=2620-train_action_mse_error=0.0000040.ckpt", "rb"), pickle_module=dill)
        
        # NEW NEW EA OBS CONFIG
        # payload = torch.load(open("data/outputs/2025.07.28/16.03.40_victor_diffusion_state_victor_diff/checkpoints/epoch=0150-train_action_mse_error=0.0001035.ckpt", "rb"), pickle_module=dill)
        # 2025 epochs with new ea config + epsilon
        # payload = torch.load(open("data/outputs/2025.07.28/16.39.25_victor_diffusion_state_victor_diff/checkpoints/latest.ckpt", "rb"), pickle_module=dill)

        # OLD NEW OBS CONFIG (Joint angles)
        # 250 epochs + image + sample
        # payload = torch.load(open("data/outputs/2025.07.29/16.18.40_victor_diffusion_image_victor_diff/checkpoints/latest.ckpt", "rb"), pickle_module=dill)
        # 500 epochs image + sample
        payload = torch.load(open("data/outputs/2025.07.29/16.18.40_victor_diffusion_image_victor_diff/checkpoints/latest.ckpt", "rb"), pickle_module=dill)

        self.cfg = payload['cfg']
        self.cfg.policy.num_inference_steps = 10
        cls = hydra.utils.get_class(self.cfg._target_)
        workspace = cls(self.cfg, output_dir=output_dir)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # get policy from workspace
        self.policy = workspace.model # type: ignore
        if self.cfg.training.use_ema:
            self.policy = workspace.ema_model # type: ignore
        self.device = self.policy.device

        self.device = torch.device(self.device)
        self.policy.to(self.device)

        # setup accumulator
        self.accumulator = ObsAccumulator(self.cfg.policy.n_obs_steps)
        # setup accumulator update timer
        self.client.timer = self.client.create_timer(
            0.1, # we trained our model on 10hz data
            self.update_accumulator_callback
        )
        # image index 
        self.imi = 0

        self.previous_act = None
    
        # self.zf = zarr.open("data/victor/victor_data_07_28_end_affector.zarr", mode='r') 
        # self.zf = zarr.open("data/victor/victor_data_07_24_single_trajectory.zarr", mode='r') 
        # self.zf = zarr.open("data/victor/victor_data_07_22_no_wrench.zarr", mode='r') 
        self.zf = zarr.open("data/victor/victor_data_07_29_all_ep_ea.zarr", mode='r') 

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
            self.client.high_freq_qos
        )

    def wrench_stop_callback(self, msg: WrenchStamped):
        w = msg.wrench
        wrench = [w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z]
        if np.abs(wrench[:3]) > 55:
            exit(-777) # exit the execution 
    
    def image_callback(self, msg: Image):
        self.latest_img = rnp.numpify(msg)
        return 

    def update_accumulator_callback(self):
        assert self.client.__getattribute__(self.side) is not None, self.side + " arm client is not initialized"

        gripper = self.arm.get_gripper_status() # type: ignore
        gripper_obs = gripper_status_to_tensor(gripper, self.client.device) # type: ignore
      
        ms = self.arm.get_motion_status().measured_joint_position
        motion_status = np.array([
            ms.joint_1, ms.joint_2, ms.joint_3, 
            ms.joint_4, ms.joint_5, ms.joint_6, 
            ms.joint_7
        ])

        if self.previous_act is not None:
            joint_cmd = self.arm.get_joint_commands()  # type: ignore
            gripper_cmd = self.arm.get_gripper_commands()
              # concat the joint commands and gripper pos requests
            curr_act = np.hstack([joint_cmd, gripper_cmd])

        else:
            joint_cmd = motion_status
            gripper_cmd = np.hstack([gripper_obs[1], gripper_obs[3], gripper_obs[5], gripper_obs[7]])
            # concat the joint commands and gripper pos requests
            curr_act = np.hstack([joint_cmd, gripper_cmd])
            self.previous_act = curr_act

        # print(wrench)

        # concat the observed joint positions, and observed gripper positions
        sim_obs = np.hstack([self.previous_act, motion_status, gripper_obs[1], gripper_obs[3], gripper_obs[5], gripper_obs[7]])
        # print(sim_obs.shape)
        print("SIM OBS:\n", sim_obs)

        self.accumulator.put({
            # "image" : np.moveaxis(self.latest_img,-1,0)/255,  # swap axis to make it fit the dataset shape
            "image" : np.moveaxis(np.array(self.zf["data/image"][self.imi]),-1,0)/255,
            "robot_obs" : sim_obs
        })

        self.data_dict.add('robot_obs', sim_obs)

        self.previous_act = curr_act
        print("previous act", self.previous_act)

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
        
    # assumes [7 dim joint angles, 4 dim gripper]
    def send_action(self, action):
        self.arm.send_joint_command(action[:7])
        self.arm.send_gripper_command(action[7:])
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
        print("s", self.arm)
        if self.arm:
            if not self.client.set_controller(self.side, 'impedance_controller', timeout=10.0):
                self.get_logger().error(f"Failed to switch {self.side} arm controller")
                return
        
        # previous_act = np.array(self.zf["data/robot_act"][0])
        # print("OLD previous_act:", self.zf["data/robot_act"][0])
        # action = self.zf["data/robot_act"][0]
        # # previous_act = action
        # self.arm.send_joint_command(action[:7])
        # self.arm.send_gripper_command(action[7:])
        # time.sleep(1)  # wait for the action to be sent
        # previous_act = None
        # Control loop
        # self.cfg.n_action_steps = 1
        for i in range(789000):  
            print('iter:', i)

            # print("ROBOT_OBS:\n", np.array(self.zf["data/robot_obs"][i]))

            np_obs_dict = dict(self.accumulator.get())
            # print("test:", self.accumulator.obs_dq)
            # device transfer
            obs_dict = dict_apply(np_obs_dict,  # type: ignore
                lambda x: torch.from_numpy(x).to(
                    device=self.device))
            
            # TODO pause the observation timer here? so that way the robot doesnt have huge gaps between but probably a bad idea
            t1 = time.time()
            # run policy
            with torch.no_grad():
                action_dict = self.policy.predict_action(obs_dict)
            t2 = time.time()
            print("inference time:", t2 - t1, "seconds")

            # device_transfer
            np_action_dict = dict_apply(action_dict,
                lambda x: x.detach().to('cpu').numpy())
            
            action = np_action_dict['action_pred']
            print(action.shape)
            
            
            # print(np_action_dict)
            # for i in range(len(action)):
            for j in range(self.cfg.n_action_steps):
                print("executing", j)
                if np.any(np.isnan(action[0][j])): continue # if there is a nan, skip command
                self.send_action(action[0][j])
                self.data_dict.add('robot_act', action[0][j])  # store the action in the data_dict
                # previous_act = action[0][j] # save the previous action for the next iteration
                self.imi = i + j
                print("action:", action[0][j])
                # print("pred action:", np_action_dict['action_pred'][0][0])
                # print("true action:", self.zf["data/robot_act"][i+j])
                time.sleep(0.1)  # wait for the action to be executed

            print("pred action:", action[0][0],"\n")
            print("true action:", self.zf["data/robot_act"][i])

        self.get_logger().info("Individual arm control example completed")

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
        store_h5_dict("data/victor/victor_run_data.h5", victor_sim.data_dict)

    except KeyboardInterrupt:
        if victor_sim:
            victor_sim.get_logger().info("Interrupted by user")
            store_h5_dict("data/victor/victor_run_data.h5", victor_sim.data_dict)
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