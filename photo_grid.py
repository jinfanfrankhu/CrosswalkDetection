#!/usr/bin/env python3
"""
photo_grid.py - Take photos with coordinate grid overlay

This script takes a single photo from the camera with a coordinate grid overlay,
similar to the one used in find_borders.py. Useful for reference when setting up
zone coordinates or documenting crosswalk layouts.
"""

import os
import argparse
import cv2
import numpy as np
import time
from datetime import datetime
from pathlib import Path

# Set environment for Wayland before other imports
os.environ['QT_QPA_PLATFORM'] = 'wayland'
if 'DISPLAY' in os.environ:
    del os.environ['DISPLAY']

# Camera imports
from picamera2 import Picamera2

# Local imports
from metasettings import (
    get_active_zone_config,
    get_system_setting,
    set_external_zone_file,
    ZONE_CONFIGURATIONS
)

class CrosswalkZone:
    """Define a zone using polygon coordinates for overlay drawing"""

    def __init__(self, name, polygon_points):
        self.name = name
        self.points = np.array(polygon_points, dtype=np.int32)

def draw_coordinate_grid(frame, grid_size=10):
    """
    Draw a coordinate grid overlay on the frame.

    Args:
        frame: OpenCV frame to draw on
        grid_size: Number of grid divisions (default 10 for 0-10 coordinates)

    Returns:
        Frame with grid overlay
    """
    height, width = frame.shape[:2]

    # Calculate grid spacing
    x_spacing = width / grid_size
    y_spacing = height / grid_size

    # Grid line color (green)
    line_color = (0, 255, 0)  # BGR format
    text_color = (0, 255, 0)

    # Draw vertical lines
    for i in range(grid_size + 1):
        x = int(i * x_spacing)
        cv2.line(frame, (x, 0), (x, height), line_color, 1)

    # Draw horizontal lines
    for j in range(grid_size + 1):
        y = int(j * y_spacing)
        cv2.line(frame, (0, y), (width, y), line_color, 1)

    # Draw intersection points and coordinate labels
    font_scale = 0.4
    thickness = 1

    for i in range(grid_size + 1):
        for j in range(grid_size + 1):
            x = int(i * x_spacing)
            y = int(j * y_spacing)

            # Skip origin point to avoid clutter
            if i == 0 and j == 0:
                continue

            # Draw intersection point (small circle)
            cv2.circle(frame, (x, y), 3, text_color, -1)

            # Create coordinate label (integer coordinates 0-10)
            label = f"({i},{j})"

            # Calculate text size for positioning
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            # Position text to avoid overlapping with the point
            text_x = x + 6
            text_y = y - 8

            # Adjust position if near edges to keep text visible
            if text_x + text_width > width:
                text_x = x - text_width - 6
            if text_y - text_height < 0:
                text_y = y + text_height + 8

            # Draw text background for better readability
            cv2.rectangle(frame,
                         (text_x - 2, text_y - text_height - 2),
                         (text_x + text_width + 2, text_y + baseline + 2),
                         (0, 0, 0), -1)

            # Draw coordinate text
            cv2.putText(frame, label, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

    return frame

def setup_zones_from_config(frame_width, frame_height):
    """
    Load zone definitions and convert to absolute coordinates for overlay drawing.

    Args:
        frame_width: Camera frame width for coordinate conversion
        frame_height: Camera frame height for coordinate conversion

    Returns:
        dict: Dictionary of zone_name -> CrosswalkZone objects
    """
    try:
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
    except Exception as e:
        print(f"Warning: Could not load zones: {e}")
        return {}

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

        # Draw zone boundary (thicker lines for photo)
        cv2.polylines(frame, [zone.points], True, color, 2)

        # Draw corner points
        for point in zone.points:
            cv2.circle(frame, tuple(point), radius=6, color=color, thickness=-1)

        # Label the zone with background
        label = f"{zone_name.upper()}"
        font_scale = 0.6
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

def add_photo_info(frame, config_name, timestamp_str):
    """
    Add photo information overlay to the frame.

    Args:
        frame: OpenCV frame to draw on
        config_name: Name of the zone configuration being used
        timestamp_str: Timestamp string for the photo

    Returns:
        Frame with info overlay
    """
    # Create info text
    info_lines = [
        f"PHOTO GRID - {config_name.upper()}",
        f"Timestamp: {timestamp_str}",
        f"Coordinate Grid: 0-10 scale"
    ]

    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (450, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    # Draw text
    y_offset = 30
    for line in info_lines:
        cv2.putText(frame, line, (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 20

    return frame

def get_output_filename(zone_config_name):
    """
    Generate output filename based on zone config and current timestamp.

    Args:
        zone_config_name: Name of the zone configuration

    Returns:
        Path to output photo file
    """
    base_dir = Path(__file__).parent / "videos" / zone_config_name
    base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"grid_photo_{timestamp}.jpg"

    return base_dir / filename

def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Take photo with coordinate grid overlay")
    parser.add_argument("--zone-config", type=str,
                       help="Zone configuration to use (built-in name or external file)")
    parser.add_argument("--resolution", type=str, default="1280x720",
                       help="Photo resolution (default: 1280x720)")
    parser.add_argument("--grid-size", type=int, default=10,
                       help="Grid divisions (default: 10 for 0-10 coordinates)")
    parser.add_argument("--show-zones", action="store_true",
                       help="Show zone boundaries on the photo")
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
    """Main function to take photo with grid overlay"""
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
    print(f"Taking photo with zone configuration: {config_name}")
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

    # Setup zones if requested
    zones = {}
    if args.show_zones:
        zones = setup_zones_from_config(width, height)
        print(f"Loaded {len(zones)} zones: {list(zones.keys())}")

    # Wait a moment for camera to stabilize
    print("Waiting for camera to stabilize...")
    time.sleep(2)

    try:
        # Capture frame
        print("Capturing photo...")
        frame = picam2.capture_array()

        # Convert from RGB to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Draw coordinate grid
        frame = draw_coordinate_grid(frame, args.grid_size)

        # Draw zone overlays if requested
        if args.show_zones:
            frame = draw_zones_on_frame(frame, zones)

        # Add photo info
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        frame = add_photo_info(frame, config_name, timestamp_str)

        # Save photo
        output_file = get_output_filename(args.zone_config or config_name)
        cv2.imwrite(str(output_file), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        print(f"Photo saved: {output_file}")
        print(f"Resolution: {width}x{height}")
        print(f"Grid divisions: {args.grid_size} (0-{args.grid_size} coordinate scale)")
        if args.show_zones:
            print(f"Zones overlaid: {len(zones)}")

    except Exception as e:
        print(f"Error capturing photo: {e}")

    finally:
        # Cleanup
        picam2.stop()
        print("Camera stopped")

if __name__ == "__main__":
    main()