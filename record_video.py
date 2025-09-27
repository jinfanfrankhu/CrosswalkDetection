#!/usr/bin/env python3
"""
record_video.py - Record video with zone overlays for crosswalk testing

This script records video from the camera with zone boundaries overlaid,
allowing you to capture test footage for different zone configurations
without needing the full detection system running.
"""

import os
import argparse
import cv2
import numpy as np
import time
from datetime import datetime
from pathlib import Path

# Local imports
from metasettings import (
    get_active_zone_config,
    get_system_setting,
    set_external_zone_file,
    ZONE_CONFIGURATIONS
)

# Set environment for Wayland before other imports
os.environ['QT_QPA_PLATFORM'] = 'wayland'
if 'DISPLAY' in os.environ:
    del os.environ['DISPLAY']

# Camera imports
from picamera2 import Picamera2

class CrosswalkZone:
    """Define a zone using polygon coordinates for overlay drawing"""

    def __init__(self, name, polygon_points):
        self.name = name
        self.points = np.array(polygon_points, dtype=np.int32)

def setup_zones_from_config(frame_width, frame_height):
    """
    Load zone definitions and convert to absolute coordinates for overlay drawing.

    Args:
        frame_width: Camera frame width for coordinate conversion
        frame_height: Camera frame height for coordinate conversion

    Returns:
        dict: Dictionary of zone_name -> CrosswalkZone objects
    """
    zone_config = get_active_zone_config()
    zone_data = zone_config['zones']

    zones = {}
    for key, value in zone_data.items():
        name = value["name"]
        norm_points = value["points"]
        abs_points = [
            (int(x * frame_width), int(y * frame_height))
            for (x, y) in norm_points
        ]
        zones[key] = CrosswalkZone(name, abs_points)

    return zones

def draw_zones_on_frame(frame, zones):
    """
    Draw zone boundaries and labels on video frame.

    Args:
        frame: OpenCV frame array to draw on
        zones: Dictionary of CrosswalkZone objects to draw

    Returns:
        Modified frame with zone overlays
    """
    if not zones:
        return frame

    # Get zone colors from settings
    zone_colors = get_system_setting('zone_colors', {})

    for zone_name, zone in zones.items():
        color = zone_colors.get(zone_name, zone_colors.get('default', (128, 128, 128)))

        # Draw zone boundary
        cv2.polylines(frame, [zone.points], True, color, 3)

        # Draw corner points
        for point in zone.points:
            cv2.circle(frame, tuple(point), radius=8, color=color, thickness=-1)

        # Label the zone with background
        label = f"{zone_name.upper()}"
        font_scale = 0.8
        thickness = 2

        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        # Position label near first point but offset to avoid overlap
        label_x = zone.points[0][0] + 10
        label_y = zone.points[0][1] - 10

        # Ensure label stays within frame bounds
        if label_y - text_height < 0:
            label_y = zone.points[0][1] + text_height + 20
        if label_x + text_width > frame.shape[1]:
            label_x = frame.shape[1] - text_width - 10

        # Draw background rectangle
        cv2.rectangle(frame,
                     (label_x - 5, label_y - text_height - 5),
                     (label_x + text_width + 5, label_y + baseline + 5),
                     (0, 0, 0), cv2.FILLED)

        # Draw text
        cv2.putText(frame, label, (label_x, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    return frame

def add_recording_info(frame, config_name, start_time):
    """
    Add recording information overlay to the frame.

    Args:
        frame: OpenCV frame to draw on
        config_name: Name of the zone configuration being used
        start_time: Time when recording started

    Returns:
        Frame with info overlay
    """
    current_time = time.time()
    elapsed_time = current_time - start_time

    # Format elapsed time as MM:SS
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    time_str = f"{minutes:02d}:{seconds:02d}"

    # Create info text
    info_lines = [
        f"RECORDING - {config_name.upper()}",
        f"Time: {time_str}",
        f"Press Ctrl+C to stop"
    ]

    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Draw text
    y_offset = 30
    for line in info_lines:
        cv2.putText(frame, line, (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 25

    return frame

def get_output_filename(zone_config_name):
    """
    Generate output filename based on zone config and current timestamp.

    Args:
        zone_config_name: Name of the zone configuration

    Returns:
        Path to output video file
    """
    base_dir = Path(__file__).parent / "videos" / zone_config_name
    base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recording_{timestamp}.mp4"

    return base_dir / filename

def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Record video with zone overlays")
    parser.add_argument("--zone-config", type=str,
                       help="Zone configuration to use (built-in name or external file)")
    parser.add_argument("--resolution", type=str, default="1280x720",
                       help="Video resolution (default: 1280x720)")
    parser.add_argument("--fps", type=int, default=30,
                       help="Frames per second (default: 30)")
    parser.add_argument("--list-configs", action="store_true",
                       help="List available built-in zone configurations")

    return parser.parse_args()

def list_available_configs():
    """List all available zone configurations"""
    print("Available built-in zone configurations:")
    print("-" * 40)

    for name, config in ZONE_CONFIGURATIONS.items():
        print(f"  {name:20} - {config['name']}")
        print(f"                       {config['description']}")
        print()

    print("External zone files in 'zones/' directory:")
    print("-" * 40)
    zones_dir = Path(__file__).parent / "zones"
    if zones_dir.exists():
        for zone_file in zones_dir.glob("*.json"):
            print(f"  {zone_file.stem}")
    else:
        print("  No zones directory found")

def main():
    """Main function to record video with zone overlays"""
    args = get_args()

    if args.list_configs:
        list_available_configs()
        return

    # Set zone configuration if specified
    if args.zone_config:
        # Check if it's a built-in config first
        if args.zone_config in ZONE_CONFIGURATIONS:
            print(f"Using built-in zone configuration: {args.zone_config}")
        else:
            # Try as external zone file
            set_external_zone_file(args.zone_config)
            print(f"Using external zone configuration: {args.zone_config}")

    # Get active zone configuration
    zone_config = get_active_zone_config()
    config_name = zone_config['name']
    print(f"Recording with zone configuration: {config_name}")
    print(f"Description: {zone_config['description']}")

    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
    except ValueError:
        print(f"Invalid resolution format: {args.resolution}")
        print("Use format like 1280x720")
        return

    # Setup camera
    print("Initializing camera...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (width, height)})
    picam2.start(config, show_preview=False)
    print(f"Camera started at {width}x{height}")

    # Setup zones based on camera resolution
    zones = setup_zones_from_config(width, height)
    print(f"Loaded {len(zones)} zones: {list(zones.keys())}")

    # Setup video writer
    output_file = get_output_filename(args.zone_config or config_name)
    print(f"Output file: {output_file}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_file), fourcc, args.fps, (width, height))

    if not out.isOpened():
        print("Error: Could not open video writer")
        picam2.stop()
        return

    print(f"Recording started at {args.fps} FPS")
    print("Press Ctrl+C to stop recording")
    print("=" * 50)

    start_time = time.time()
    frame_count = 0

    try:
        while True:
            # Capture frame
            frame = picam2.capture_array()

            # Convert from RGB to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Draw zone overlays
            frame = draw_zones_on_frame(frame, zones)

            # Add recording info
            frame = add_recording_info(frame, config_name, start_time)

            # Write frame to video file
            out.write(frame)
            frame_count += 1

            # Print status every 5 seconds
            if frame_count % (args.fps * 5) == 0:
                elapsed = time.time() - start_time
                print(f"Recording... {frame_count} frames, {elapsed:.1f}s elapsed")

            # Small delay to maintain frame rate
            time.sleep(1.0 / args.fps)

    except KeyboardInterrupt:
        print("\nStopping recording...")

    finally:
        # Cleanup
        total_time = time.time() - start_time
        print(f"\nRecording completed:")
        print(f"  Duration: {total_time:.1f} seconds")
        print(f"  Frames: {frame_count}")
        print(f"  Average FPS: {frame_count / total_time:.1f}")
        print(f"  Output file: {output_file}")

        out.release()
        picam2.stop()
        print("Camera stopped")

if __name__ == "__main__":
    main()