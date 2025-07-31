# this sim test simply takes input that it was trained on 
# and "plays back" the inferred actions

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
from std_msgs.msg import String
from geometry_msgs.msg import TransformStamped
from victor_hardware_interfaces.msg import (
    MotionStatus, 
    Robotiq3FingerCommand, 
    Robotiq3FingerStatus,
    JointValueQuantity,
    Robotiq3FingerActuatorCommand,
    Robotiq3FingerActuatorStatus
)
from diffusion_policy.common.victor_accumulator import ObsAccumulator
import zarr
from victor_python.victor_policy_client import VictorPolicyClient
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.workspace.base_workspace import BaseWorkspace

from data_utils import SmartDict, store_h5_dict, store_zarr_dict
from victor_python.victor_utils import wrench_to_tensor, gripper_status_to_tensor

class VictorSimClient:
    def __init__(self, device: Union[str, torch.device] = 'cpu'):
        # Initialize client with both arms enabled and specified device
        self.side = 'right'
        self.client = VictorPolicyClient('policy_example', enable_left = self.side == 'left',
                                            enable_right = self.side == 'right', device=device)
        self.arm = self.client.right if self.side == 'right' else self.client.left
        self.get_logger = self.client.get_logger
        # self.device = torch.device(device)    # use model device

        self.data_dict = SmartDict()
        ### SETUP POLICY
        output_dir = "data/victor_eval_output"
        # if os.path.exists(output_dir):
        #     click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

         # load checkpoint
        # NEW OBS CONFIG TODO
        # 50 epoch state only
        # payload = torch.load(open("data/outputs/2025.07.22/13.15.22_victor_diffusion_state_victor_diff/checkpoints/latest.ckpt", "rb"), pickle_module=dill)
        # 210 epoch image + state
        # payload = torch.load(open("data/outputs/2025.07.22/17.14.10_victor_diffusion_image_victor_diff/checkpoints/epoch=0180-train_action_mse_error=0.0000083.ckpt", "rb"), pickle_module=dill)
        # 100 epoch image + state + epsilon
        # payload = torch.load(open("data/outputs/2025.07.23/17.25.58_victor_diffusion_image_victor_diff/checkpoints/epoch=0100-train_action_mse_error=0.0002452.ckpt", "rb"), pickle_module=dill)
        # 2620 epoch state only
        # payload = torch.load(open("data/outputs/2025.07.24/16.59.38_victor_diffusion_state_victor_diff/checkpoints/epoch=2620-train_action_mse_error=0.0000040.ckpt", "rb"), pickle_module=dill)
        # 555 epoch state only -> SINGLE OBS STEP
        # payload = torch.load(open("data/outputs/2025.07.25/12.16.22_victor_diffusion_state_victor_diff/checkpoints/epoch=0555-train_action_mse_error=0.0001751.ckpt", "rb"), pickle_module=dill)


        # NEW NEW EA OBS CONFIG
        # payload = torch.load(open("data/outputs/2025.07.28/16.03.40_victor_diffusion_state_victor_diff/checkpoints/epoch=0150-train_action_mse_error=0.0001035.ckpt", "rb"), pickle_module=dill)
        # 2025 epochs with new ea config + epsilon
        # payload = torch.load(open("data/outputs/2025.07.28/16.39.25_victor_diffusion_state_victor_diff/checkpoints/latest.ckpt", "rb"), pickle_module=dill)
        # 3000 epochs + sample
        # payload = torch.load(open("data/outputs/2025.07.28/21.28.14_victor_diffusion_state_victor_diff/checkpoints/epoch=2900-train_action_mse_error=0.0000000.ckpt", "rb"), pickle_module=dill)

        # OLD NEW OBS CONFIG (Joint angles)
        # 250 epochs + image + sample
        # payload = torch.load(open("data/outputs/2025.07.29/16.18.40_victor_diffusion_image_victor_diff/checkpoints/latest.ckpt", "rb"), pickle_module=dill)
        # 500 epochs image + sample
        payload = torch.load(open("data/outputs/2025.07.29/16.18.40_victor_diffusion_image_victor_diff/checkpoints/latest.ckpt", "rb"), pickle_module=dill)

        # VALIDATION MASK: [False  True False False False False  True False False  True  True False False False False]

        self.cfg = payload['cfg']
        cfg = self.cfg
        cfg.policy.num_inference_steps = 10
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

        # setup accumulator
        self.accumulator = ObsAccumulator(cfg.policy.n_obs_steps)
      
        # self.zf = zarr.open("data/victor/victor_data_07_22_no_wrench.zarr", mode='r') 
        # self.zf = zarr.open("data/victor/victor_data_07_24_single_trajectory.zarr", mode='r') 
        # self.zf = zarr.open("data/victor/victor_data_07_28_end_affector.zarr", mode='r') 
        self.zf = zarr.open("data/victor/victor_data_07_29_all_ep_ea.zarr", mode='r') 

        print(self.zf["meta/episode_name"])
    
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
        self.get_logger().info("Starting individual arm contrvictor_policy_bridgeol example...")
        
        if not self.wait_for_server(10.0):
            self.get_logger().error("Failed to connect to server")
            return
        
        # individual arm control
        if self.arm:
            if not self.client.set_controller(self.side, 'impedance_controller', timeout=10.0):
                self.get_logger().error(f"Failed to switch {self.side} arm controller")
                return
        
        print(self.client.__getattribute__(self.side))
        previous_act = self.zf["data/robot_act"][0]
        # Control loop
        self.cfg.n_action_steps = 1
        for i in range(741, 1639, self.cfg.n_action_steps):  #789 10535, 11193
            print('iter:', i)
            # get observations
            right_pos = self.arm.get_joint_positions() # type: ignore
            rms = self.arm.get_motion_status().commanded_joint_position
            right_motion_status = np.array([
                rms.joint_1, rms.joint_2, rms.joint_3, 
                rms.joint_4, rms.joint_5, rms.joint_6, 
                rms.joint_7
            ])

            # print(self.arm.get_motion_status().commanded_joint_position.get_fields_and_field_types())
            right_gripper = self.arm.get_gripper_status() # type: ignore
            gripper_obs = gripper_status_to_tensor(right_gripper, self.client.device) # type: ignore
            sim_obs = np.hstack([previous_act, right_motion_status, gripper_obs[1], gripper_obs[3], gripper_obs[5], gripper_obs[7]])
            self.data_dict.add('robot_obs', sim_obs)

            if right_pos is not None:
                self.accumulator.put({
                    "image" : np.moveaxis(np.array(self.zf["data/image"][i]),-1,0)/255,  # swap axis to make it fit the dataset shape
                    "robot_obs" : np.array(self.zf["data/robot_obs"][i])
                })

                print("OBS:\n", np.array(self.zf["data/robot_obs"][i]))

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
                
                action = np_action_dict['action_pred']
                print(action.shape)
                
              
                # print(np_action_dict)
                # for i in range(len(action)):
                for j in range(self.cfg.n_action_steps):
                    print("executing", j)
                    self.send_action(action[0][j])
                    self.data_dict.add('robot_act', action[0][j])  # store the action in the data_dict
                    previous_act = action[0][j] # save the previous action for the next iteration

                    print("action:", action[0][j])
                    # print("pred action:", np_action_dict['action_pred'][0][0])
                    print("true action:", self.zf["data/robot_act"][i+j])
                    time.sleep(0.1)  # wait for the action to be executed

               

                # action = np.array(self.zf["data/robot_act"][i])
                # previous_act = action # save the previous action for the next iteration
                # # print(action[:7])
                # self.arm.send_joint_command(action[:7])
                # self.arm.send_gripper_command(action[7:])
                # time.sleep(0.1)


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
        store_h5_dict("data/victor/victor_playback_data.h5", victor_sim.data_dict)
        store_zarr_dict("data/victor/victor_playback_data.zarr.zip", victor_sim.data_dict)

    except KeyboardInterrupt:
        if victor_sim:
            victor_sim.get_logger().info("Interrupted by user")
            store_h5_dict("data/victor/victor_playback_data.h5", victor_sim.data_dict)
            store_zarr_dict("data/victor/victor_playback_data.zarr.zip", victor_sim.data_dict)
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