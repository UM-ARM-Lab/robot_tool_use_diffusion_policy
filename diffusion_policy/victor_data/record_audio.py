#!/usr/bin/env python3
"""
record_audio.py

Record audio from the default input device and save to a WAV file.
Usage:
    python record_audio.py /path/to/output_folder [--samplerate SR] [--channels CH]

Press Ctrl+C to stop recording.
"""

import argparse
import os
import sys
from datetime import datetime

import sounddevice as sd
import soundfile as sf

def parse_args():
    p = argparse.ArgumentParser(description="Record audio and save to a folder.")
    p.add_argument("output_folder",
                   help="Directory where recordings will be saved")
    p.add_argument("--samplerate", "-r", type=int, default=44100,
                   help="Sampling rate in Hz (default: 44100)")
    p.add_argument("--channels", "-c", type=int, default=1,
                   help="Number of input channels (default: 1 mono)")
    return p.parse_args()

def main():
    args = parse_args()

    # Ensure output directory exists
    try:
        os.makedirs(args.output_folder, exist_ok=True)
    except Exception as e:
        print(f"Error creating output folder: {e}", file=sys.stderr)
        sys.exit(1)

    # Build filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(args.output_folder, f"recording_{timestamp}.wav")

    print(f"Recording to: {filename}")
    print("Press Ctrl+C to stop.")

    # Open soundfile and stream InputStream
    try:
        with sf.SoundFile(filename, mode='w',
                          samplerate=args.samplerate,
                          channels=args.channels,
                          subtype='PCM_16') as file:
            with sd.InputStream(samplerate=args.samplerate,
                                channels=args.channels,
                                callback=lambda indata, frames, time, status: file.write(indata)):
                while True:
                    sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
    except Exception as e:
        print(f"\nError during recording: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
