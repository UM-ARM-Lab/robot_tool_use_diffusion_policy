#!/usr/bin/env python3

import argparse
import os
import re
import subprocess
import signal
import datetime
import time
import shutil
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from std_msgs.msg import Float64MultiArray
from victor_hardware_interfaces.msg import Robotiq3FingerCommand, Robotiq3FingerStatus

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rosbag2_py._storage import StorageFilter
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# This script is almost identical to record_victor_demo.py
# but implements replay functionality with these steps:
# - takes in --replay path, and finds the replay_path/rosbag as the rosbag to replay
# - tag is cap_replay_<orig_tag>
#
# 1. programmatically reads the first message of /right_arm_impedance_controller/commands and /victor/right/gripper_status from the bag
# 2. publishes each message 10 times at 0.1 second intervals (converts gripper status to command)
# 3. wait for 5 seconds
# 4. start main recording processes

def default_gripper_command():
    """Create a default gripper command with standard settings"""
    cmd = Robotiq3FingerCommand()
    cmd.finger_a_command.speed = 0.5
    cmd.finger_b_command.speed = 0.5
    cmd.finger_c_command.speed = 0.5
    cmd.scissor_command.speed = 1.0

    cmd.finger_a_command.force = 1.0
    cmd.finger_b_command.force = 1.0
    cmd.finger_c_command.force = 1.0
    cmd.scissor_command.force = 1.0

    cmd.scissor_command.position = 1.0
    return cmd

def get_gripper_closed_fraction_msg(position: float, scissor_position: float = 0.2):
    """
    Args:
        position: 0.0 is open, 1.0 is closed
        scissor_position: 0.0 is wide, 1.0 is narrow
    """
    msg = default_gripper_command()
    msg.finger_a_command.position = position
    msg.finger_b_command.position = position
    msg.finger_c_command.position = position
    msg.scissor_command.position = scissor_position
    return msg

def status_to_command(gripper_status: Robotiq3FingerStatus) -> Robotiq3FingerCommand:
    """Convert gripper status to gripper command"""
    # Use the average finger position as the closed fraction
    return get_gripper_closed_fraction_msg(
        gripper_status.finger_a_status.position,
        gripper_status.scissor_status.position
    )

def first_msg_of_type(bag_path: str, type_name: str, topic_name: str | None = None):
    """
    bag_path: path to the directory containing metadata.yaml
    type_name: e.g. "sensor_msgs/msg/Image" or "geometry_msgs/msg/Twist"
    topic_name: specific topic name to filter for (optional, if None will use any topic with the type)
    returns: (topic_name, timestamp_ns, msg) or (None, None, None) if not found
    """

    # Detect storage; Humble defaults to sqlite3 unless you recorded with mcap
    storage_id = 'sqlite3'  # change to 'mcap' if your bag is MCAP
    storage_options = StorageOptions(uri=bag_path, storage_id=storage_id)
    converter_options = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )

    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    # Find topics that have the requested type and optionally specific topic name
    topics_and_types = reader.get_all_topics_and_types()
    if topic_name is not None:
        # Filter for specific topic name AND type
        wanted_topics = [t.name for t in topics_and_types if t.type == type_name and t.name == topic_name]
    else:
        # Filter for any topic with the requested type
        wanted_topics = [t.name for t in topics_and_types if t.type == type_name]
    
    if not wanted_topics:
        return None, None, None  # no topic in this bag has that type/name combination

    # Optionally filter the reader to those topics (faster)
    reader.set_filter(StorageFilter(topics=wanted_topics))

    # Prepare deserializer
    MsgType = get_message(type_name)

    # Iterate until we find the first message
    while reader.has_next():
        found_topic_name, serialized_data, timestamp = reader.read_next()
        msg = deserialize_message(serialized_data, MsgType)
        return found_topic_name, timestamp, msg

    return None, None, None  # bag had none of that type


class ReplayCommandPublisher(Node):
    def __init__(self):
        super().__init__('replay_command_publisher')
        
        # QoS profile for reliable communication
        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        
        # Publisher for right arm impedance controller commands
        self.arm_command_publisher = self.create_publisher(
            Float64MultiArray,
            '/right_arm_impedance_controller/commands',
            qos_profile
        )
        
        # Publisher for gripper commands
        self.gripper_command_publisher = self.create_publisher(
            Robotiq3FingerCommand,
            '/victor/right_arm/gripper_command',
            qos_profile
        )
        
        self.get_logger().info('Replay command publisher initialized')

    def publish_commands_repeatedly(self, arm_msg, gripper_msg, count=10, interval=0.2):
        """Publish both commands repeatedly with the specified interval"""
        for i in range(count):
            if arm_msg is not None:
                self.arm_command_publisher.publish(arm_msg)
            if gripper_msg is not None:
                self.gripper_command_publisher.publish(gripper_msg)
            # self.get_logger().info(f'Published command set {i+1}/{count}')
            
            if i < count - 1:  # Don't sleep after the last iteration
                time.sleep(interval)

def main():
    parser = argparse.ArgumentParser(description="Run Zivid capture and rosbag record with replay functionality.")
    parser.add_argument("--replay", required=True, help="Path to the replay directory containing rosbag")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio recording")
    args = parser.parse_args()

    # Validate replay path
    replay_path = Path(args.replay)
    if not replay_path.exists():
        raise FileNotFoundError(f"Replay path does not exist: {replay_path}")
    
    replay_rosbag_path = replay_path / "rosbag"
    if not replay_rosbag_path.exists():
        raise FileNotFoundError(f"Rosbag directory not found in replay path: {replay_rosbag_path}")

    # Parse tag from replay directory name
    replay_dir_name = replay_path.name
    # Match pattern: yyyymmdd_hhmmss_{tag}
    pattern = r'^\d{8}_\d{6}_(.+)$'
    match = re.match(pattern, replay_dir_name)
    if not match:
        raise ValueError(f"Replay directory name '{replay_dir_name}' does not match expected format 'yyyymmdd_hhmmss_{{tag}}'")
    
    original_tag = match.group(1)
    print(f"Parsed original tag: {original_tag}")

    # Construct output directory with replayof prefix
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = Path.home() / "datasets" / "robotool" / f"{timestamp}_replayof_{original_tag}"
    os.makedirs(root_dir, exist_ok=True)

    # Load environment variables
    ros_ws_path = os.environ.get("ROS_WS_PATH")
    if not ros_ws_path:
        raise EnvironmentError("Environment variable $ROS_WS_PATH is not set.")

    print(f"Replay setup with output root: {root_dir}")
    print(f"Replaying from: {replay_rosbag_path}")

    # Prompt user to switch to impedance controllers
    print("\n" + "="*60)
    print("IMPORTANT: Please ensure the robot is in impedance control mode")
    print("Switch to left/right arm impedance controllers before proceeding.")
    print("="*60)
    
    response = input("Have you switched to impedance controllers? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Please switch to impedance controllers and run the script again.")
        return

    # Initialize ROS 2
    rclpy.init()
    
    # Create the command publisher node
    command_publisher = ReplayCommandPublisher()
    
    # Read first messages from the bag file
    # print("[INFO] Reading first arm command from bag...")
    arm_topic, arm_timestamp, arm_msg = first_msg_of_type(
        str(replay_rosbag_path), 
        "std_msgs/msg/Float64MultiArray",
        "/right_arm_impedance_controller/commands"
    )
    print(arm_topic, arm_timestamp, arm_msg)
    
    # print("[INFO] Reading first gripper status from bag...")
    gripper_topic, gripper_timestamp, gripper_status_msg = first_msg_of_type(
        str(replay_rosbag_path), 
        "victor_hardware_interfaces/msg/Robotiq3FingerStatus",
        "/victor/right_arm/gripper_status"
    )
    
    # Convert gripper status to command
    gripper_msg = None
    if gripper_status_msg is not None:
        gripper_msg = status_to_command(gripper_status_msg)

    if arm_msg is None or gripper_msg is None:
        print("[ERROR] No commands found in bag file")
        command_publisher.destroy_node()
        rclpy.shutdown()
        return
    
    # Publish commands 25 times at 0.2 second intervals
    print("[INFO] Publishing commands 25 times at 0.2 second intervals...")
    command_publisher.publish_commands_repeatedly(arm_msg, gripper_msg, count=25, interval=0.2)
    
    # Commands for main recording
    zivid_cmd = [
        "python3",
        f"{ros_ws_path}/src/arm_zivid/arm_zivid/arm_zivid_ros_node_local.py",
        "--verbose",
        "-s", f"{ros_ws_path}/config/zivid2_11Hz_Engine_2D3D_Final.yml",
        "-n", "zivid",
        "-r", str(root_dir),
        "--output-format", "h5",
    ]

    rosbag_cmd = [
        "ros2", "bag", "record", "-a",
        "--output", str(root_dir / "rosbag")
    ]

    # Replay command for main execution
    replay_cmd = [
        "ros2", "bag", "play", str(replay_rosbag_path),
        "--topics", 
        "/left_arm_impedance_controller/commands",
        "/victor/left_arm/gripper_command",
        "/right_arm_impedance_controller/commands",
        "/victor/right_arm/gripper_command"
    ]

    audio_cmd = [
        "python3",
        "diffusion_policy/victor_data/record_audio.py",
        str(root_dir),
        "--samplerate", "44100",
        "--channels", "1"
    ]

    print("Starting main recording processes...")

    # Start subprocesses
    zivid_proc = subprocess.Popen(zivid_cmd, preexec_fn=os.setsid)
    rosbag_proc = subprocess.Popen(rosbag_cmd, preexec_fn=os.setsid)
    
    audio_proc = None
    if not args.no_audio:
        audio_proc = subprocess.Popen(audio_cmd, preexec_fn=os.setsid)

    # Wait 2 seconds before starting replay
    print("[INFO] Waiting 2 seconds before starting main replay...")
    time.sleep(2.0)
    
    # Start main rosbag replay
    print("[INFO] Starting main rosbag replay...")
    replay_proc = subprocess.Popen(replay_cmd, preexec_fn=os.setsid)

    try:
        # Wait for replay to finish first
        replay_proc.wait()
        print("[INFO] Replay finished. Terminating other processes...")
        
        # After replay finishes, terminate the other processes
        processes = [zivid_proc, rosbag_proc]
        if audio_proc:
            processes.append(audio_proc)
            
        for proc in processes:
            try:
                os.killpg(proc.pid, signal.SIGINT)
            except Exception:
                pass

        # Wait for all processes to finish
        for proc in processes:
            try:
                proc.wait()
            except Exception:
                pass
                
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received. Terminating all processes...")

        # Send SIGINT to all process groups so they cleanly shutdown
        processes = [zivid_proc, rosbag_proc, replay_proc]
        if audio_proc:
            processes.append(audio_proc)
            
        for proc in processes:
            try:
                os.killpg(proc.pid, signal.SIGINT)
            except Exception:
                pass

        for proc in processes:
            try:
                proc.wait()
            except Exception:
                pass

    # Cleanup ROS - check if it's still initialized before shutting down
    try:
        command_publisher.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    except Exception as e:
        print(f"[WARNING] Error during ROS cleanup: {e}")

    print("[INFO] All processes have exited.")

    # Copy annotation.json from original folder if it exists
    annotation_source = replay_path / "annotation.json"
    annotation_dest = root_dir / "annotation.json"
    
    if annotation_source.exists():
        try:
            shutil.copy2(annotation_source, annotation_dest)
            print(f"[INFO] Copied annotation.json from {annotation_source} to {annotation_dest}")
        except Exception as e:
            print(f"[WARNING] Failed to copy annotation.json: {e}")
    else:
        print(f"[WARNING] No annotation.json found in replay directory: {annotation_source}")

    # Launch video generation process
    zivid_dir = root_dir / "zivid"
    if zivid_dir.exists():
        print(f"[INFO] Launching video generation for {zivid_dir}...")
        video_cmd = [
            "python3",
            f"{ros_ws_path}/src/arm_zivid/arm_zivid/zivid_ds_to_vid.py",
            str(zivid_dir),
            "--data-type", "rgb",
            "--fps", "10.0"
        ]
        
        try:
            subprocess.run(video_cmd, check=True)
            print("[INFO] Video generation completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[WARNING] Video generation failed with return code {e.returncode}")
        except Exception as e:
            print(f"[WARNING] Error during video generation: {e}")
    else:
        print(f"[WARNING] Zivid directory not found: {zivid_dir}")

if __name__ == "__main__":
    main()
