#!/usr/bin/env python3

import argparse
import os
import subprocess
import signal
import datetime
from pathlib import Path
def main():
    parser = argparse.ArgumentParser(description="Run Zivid capture and rosbag record in parallel.")
    parser.add_argument("--tag", required=True, help="Tag to append to the dataset folder")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio recording")
    args = parser.parse_args()

    # Construct output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = Path.home() / "datasets" / "robotool" / f"{timestamp}_{args.tag}"
    os.makedirs(root_dir, exist_ok=True)

    # Commands
    # Load environment variables
    ros_ws_path = os.environ.get("ROS_WS_PATH")
    if not ros_ws_path:
        raise EnvironmentError("Environment variable $ROS_WS_PATH is not set.")

    zivid_cmd = [
        "python3",
        f"{ros_ws_path}/src/arm_zivid/arm_zivid/arm_zivid_ros_node_local.py",
        "--verbose",
        "-s", f"{ros_ws_path}/config/zivid2_Settings_Zivid_Two_M70_ParcelsMatte_10Hz_4xsparse_enginetop_boxed.yml",
        "-n", "zivid",
        "-r", str(root_dir),
        "--output-format", "h5",
    ]

    rosbag_cmd = [
        "ros2", "bag", "record", "-a",
        "--output", str(root_dir / "rosbag")
    ]

    audio_cmd = [
        "python3",
        f"diffusion_policy/victor_data/record_audio.py",
        str(root_dir),
        "--samplerate", "44100",
        "--channels", "1"
    ]

    print(f"Launching processes with output root: {root_dir}")

    # Start subprocesses
    zivid_proc = subprocess.Popen(zivid_cmd, preexec_fn=os.setsid)
    rosbag_proc = subprocess.Popen(rosbag_cmd, preexec_fn=os.setsid)
    
    audio_proc = None
    if not args.no_audio:
        audio_proc = subprocess.Popen(audio_cmd, preexec_fn=os.setsid)

    try:
        # Wait for all to complete
        zivid_proc.wait()
        rosbag_proc.wait()
        if audio_proc:
            audio_proc.wait()
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received. Terminating all processes...")

        # Send SIGINT to all process groups so they cleanly shutdown
        os.killpg(zivid_proc.pid, signal.SIGINT)
        os.killpg(rosbag_proc.pid, signal.SIGINT)
        if audio_proc:
            os.killpg(audio_proc.pid, signal.SIGINT)

        zivid_proc.wait()
        rosbag_proc.wait()
        if audio_proc:
            audio_proc.wait()

    print("[INFO] All processes have exited.")

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
            video_proc = subprocess.run(video_cmd, check=True)
            print("[INFO] Video generation completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[WARNING] Video generation failed with return code {e.returncode}")
        except Exception as e:
            print(f"[WARNING] Error during video generation: {e}")
    else:
        print(f"[WARNING] Zivid directory not found: {zivid_dir}")

if __name__ == "__main__":
    main()
