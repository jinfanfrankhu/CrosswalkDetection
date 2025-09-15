"""
Metasettings for Crosswalk Detection System

This file contains all configuration settings for the crosswalk detection system,
including zone definitions, detection parameters, and system settings.
"""

import os
import json
import math
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / "logs"
ASSETS_DIR = BASE_DIR / "assets"
ZONES_DIR = BASE_DIR / "zones"

# Ensure directories exist
LOGS_DIR.mkdir(exist_ok=True)
ASSETS_DIR.mkdir(exist_ok=True)

# System Settings
SYSTEM_CONFIG = {
    # Camera and AI Model Settings
    "imx_model_path": "/usr/share/imx500-models/imx500_network_nanodet_plus_416x416_pp.rpk",
    "labels_file": str(ASSETS_DIR / "coco_labels.txt"),
    
    # Detection Parameters
    "detection_threshold": 0.55,
    "iou_threshold": 0.65,
    "max_detections": 10,
    
    # Web Server Settings
    "web_port": 8080,
    "web_stream_fps": 10,
    "jpeg_quality": 85,
    
    # Tracking Settings
    "max_disappeared_frames": 10,
    "max_crossing_events": 1000,
    
    # Relevant object types for tracking
    "tracked_objects": ['person', 'car', 'truck', 'bus', 'bicycle', 'motorcycle'],
    
    # Zone visualization colors (BGR format for OpenCV)
    "zone_colors": {
        'crosswalk': (0, 255, 255),      # Yellow
        'north_lane': (255, 0, 0),       # Blue
        'south_lane': (0, 0, 255),       # Red
        'east_lane': (0, 255, 0),        # Green
        'west_lane': (150, 75, 0),       # Brown
        'default': (128, 128, 128)       # Gray
    },
    
    # Object detection colors (BGR format for OpenCV)
    "detection_colors": {
        'person': (0, 255, 0),           # Green
        'car': (255, 0, 0),              # Blue
        'truck': (255, 0, 0),            # Blue
        'bus': (255, 0, 0),              # Blue
        'bicycle': (0, 255, 255),        # Yellow
        'motorcycle': (0, 255, 255),     # Yellow
        'default': (255, 255, 255)       # White
    }
}

# Zone Configurations
# Define multiple zone configurations for different locations/scenarios
ZONE_CONFIGURATIONS = {
    # Test configuration - simple crosswalk setup
    "test": {
        "name": "Test Crosswalk Configuration",
        "description": "Basic crosswalk setup for testing",
        "zones": {
            "crosswalk": {
                "name": "crossing_zone",
                "points": [
                    [0.2, 0.4],
                    [0.8, 0.4],
                    [0.8, 0.6],
                    [0.2, 0.6]
                ]
            },
            "south_lane": {
                "name": "vehicle_lane_south",
                "points": [
                    [0.1, 0.58],
                    [0.9, 0.58],
                    [0.9, 0.9],
                    [0.1, 0.9]
                ]
            }
        }
    },
    
    # More comprehensive 4-way intersection
    "intersection": {
        "name": "4-Way Intersection Configuration",
        "description": "Complete 4-way intersection with crosswalks and vehicle lanes",
        "zones": {
            "crosswalk_ns": {
                "name": "north_south_crosswalk",
                "points": [
                    [0.4, 0.1],
                    [0.6, 0.1],
                    [0.6, 0.9],
                    [0.4, 0.9]
                ]
            },
            "crosswalk_ew": {
                "name": "east_west_crosswalk", 
                "points": [
                    [0.1, 0.4],
                    [0.9, 0.4],
                    [0.9, 0.6],
                    [0.1, 0.6]
                ]
            },
            "north_lane": {
                "name": "vehicle_lane_north",
                "points": [
                    [0.1, 0.1],
                    [0.4, 0.1],
                    [0.4, 0.4],
                    [0.1, 0.4]
                ]
            },
            "south_lane": {
                "name": "vehicle_lane_south",
                "points": [
                    [0.6, 0.6],
                    [0.9, 0.6],
                    [0.9, 0.9],
                    [0.6, 0.9]
                ]
            },
            "east_lane": {
                "name": "vehicle_lane_east",
                "points": [
                    [0.6, 0.1],
                    [0.9, 0.1],
                    [0.9, 0.4],
                    [0.6, 0.4]
                ]
            },
            "west_lane": {
                "name": "vehicle_lane_west",
                "points": [
                    [0.1, 0.6],
                    [0.4, 0.6],
                    [0.4, 0.9],
                    [0.1, 0.9]
                ]
            }
        }
    },
    
    # Simple single crosswalk
    "simple_crosswalk": {
        "name": "Simple Crosswalk Configuration",
        "description": "Single crosswalk with approach zones",
        "zones": {
            "crosswalk": {
                "name": "pedestrian_crossing",
                "points": [
                    [0.0, 0.45],
                    [1.0, 0.45],
                    [1.0, 0.55],
                    [0.0, 0.55]
                ]
            },
            "approach_north": {
                "name": "approach_zone_north",
                "points": [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 0.45],
                    [0.0, 0.45]
                ]
            },
            "approach_south": {
                "name": "approach_zone_south",
                "points": [
                    [0.0, 0.55],
                    [1.0, 0.55],
                    [1.0, 1.0],
                    [0.0, 1.0]
                ]
            }
        }
    }
}

# Active Configuration
# Change this to switch between different zone configurations
ACTIVE_ZONE_CONFIG = "test"  # Options: "test", "intersection", "simple_crosswalk"
EXTERNAL_ZONE_FILE = None  # Set to load from external JSON file instead

def order_polygon_points(points):
    """
    Order polygon points to form a proper convex polygon (counter-clockwise).

    Args:
        points: List of [x, y] coordinate pairs

    Returns:
        List of [x, y] coordinates ordered counter-clockwise
    """
    if len(points) < 3:
        return points

    # Find the centroid
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)

    # Sort points by angle from centroid
    def angle_from_centroid(point):
        return math.atan2(point[1] - cy, point[0] - cx)

    sorted_points = sorted(points, key=angle_from_centroid)

    return sorted_points

def validate_and_fix_polygon(points):
    """
    Validate and fix polygon points to ensure they form a proper convex polygon.

    Args:
        points: List of [x, y] coordinate pairs

    Returns:
        List of [x, y] coordinates that form a valid polygon
    """
    if len(points) < 3:
        raise ValueError(f"Polygon must have at least 3 points, got {len(points)}")

    # Order the points properly
    ordered_points = order_polygon_points(points)

    # Validate that coordinates are in range [0, 1]
    for point in ordered_points:
        if not (0 <= point[0] <= 1 and 0 <= point[1] <= 1):
            raise ValueError(f"Point coordinates must be between 0 and 1, got {point}")

    return ordered_points

def load_zone_file(zone_filename):
    """
    Load zone configuration from a JSON file in the zones directory.

    Args:
        zone_filename: Name of the zone file (with or without .json extension)

    Returns:
        dict: Zone configuration in the expected format
    """
    # Add .json extension if not present
    if not zone_filename.endswith('.json'):
        zone_filename += '.json'

    zone_file_path = ZONES_DIR / zone_filename

    if not zone_file_path.exists():
        raise FileNotFoundError(f"Zone file not found: {zone_file_path}")

    with open(zone_file_path, 'r') as f:
        zones_data = json.load(f)

    # Fix polygon point ordering for all zones
    fixed_zones_data = {}
    for zone_key, zone_info in zones_data.items():
        fixed_zone_info = zone_info.copy()
        if 'points' in zone_info:
            try:
                fixed_zone_info['points'] = validate_and_fix_polygon(zone_info['points'])
                print(f"Fixed polygon ordering for zone '{zone_key}'")
            except ValueError as e:
                print(f"Warning: Could not fix polygon for zone '{zone_key}': {e}")
                fixed_zone_info['points'] = zone_info['points']  # Keep original if fix fails
        fixed_zones_data[zone_key] = fixed_zone_info

    # Convert to the expected format
    zone_config = {
        "name": f"External Zone Config: {zone_filename}",
        "description": f"Loaded from {zone_filename}",
        "zones": fixed_zones_data
    }

    return zone_config

def set_external_zone_file(zone_filename):
    """
    Set the external zone file to use instead of built-in configurations.

    Args:
        zone_filename: Name of the zone file to load
    """
    global EXTERNAL_ZONE_FILE
    EXTERNAL_ZONE_FILE = zone_filename

def fix_builtin_zone_config(config):
    """
    Fix polygon point ordering for a built-in zone configuration.

    Args:
        config: Zone configuration dictionary

    Returns:
        Fixed zone configuration with properly ordered polygon points
    """
    fixed_config = {
        "name": config["name"],
        "description": config["description"],
        "zones": {}
    }

    for zone_key, zone_info in config["zones"].items():
        fixed_zone_info = zone_info.copy()
        if 'points' in zone_info:
            try:
                fixed_zone_info['points'] = validate_and_fix_polygon(zone_info['points'])
            except ValueError as e:
                print(f"Warning: Could not fix polygon for built-in zone '{zone_key}': {e}")
                fixed_zone_info['points'] = zone_info['points']  # Keep original if fix fails
        fixed_config["zones"][zone_key] = fixed_zone_info

    return fixed_config

def get_active_zone_config():
    """
    Get the currently active zone configuration.

    Returns:
        dict: The active zone configuration
    """
    # If external zone file is set, load from that instead
    if EXTERNAL_ZONE_FILE is not None:
        return load_zone_file(EXTERNAL_ZONE_FILE)

    # Otherwise use built-in configuration
    if ACTIVE_ZONE_CONFIG not in ZONE_CONFIGURATIONS:
        raise ValueError(f"Active zone config '{ACTIVE_ZONE_CONFIG}' not found in ZONE_CONFIGURATIONS")

    # Fix polygon ordering for built-in config too
    return fix_builtin_zone_config(ZONE_CONFIGURATIONS[ACTIVE_ZONE_CONFIG])

def get_log_path():
    """
    Get the log file path based on the active zone configuration.

    Returns:
        Path: Path to the log file
    """
    if EXTERNAL_ZONE_FILE is not None:
        # Use external zone filename for log
        base_name = Path(EXTERNAL_ZONE_FILE).stem
        return LOGS_DIR / f"{base_name}.jsonl"
    else:
        return LOGS_DIR / f"{ACTIVE_ZONE_CONFIG}.jsonl"

def get_system_setting(key, default=None):
    """
    Get a system configuration setting.
    
    Args:
        key: Setting key to retrieve
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    return SYSTEM_CONFIG.get(key, default)

def list_available_configs():
    """
    List all available zone configurations.
    
    Returns:
        list: List of configuration names with descriptions
    """
    configs = []
    for name, config in ZONE_CONFIGURATIONS.items():
        configs.append({
            'name': name,
            'title': config['name'],
            'description': config['description']
        })
    return configs

# Validation functions
def validate_zone_config(config_name):
    """
    Validate that a zone configuration is properly formatted.
    
    Args:
        config_name: Name of the configuration to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if config_name not in ZONE_CONFIGURATIONS:
        return False
        
    config = ZONE_CONFIGURATIONS[config_name]
    
    # Check required top-level keys
    if not all(key in config for key in ['name', 'description', 'zones']):
        return False
        
    # Check each zone
    for zone_name, zone_data in config['zones'].items():
        if not all(key in zone_data for key in ['name', 'points']):
            return False
        
        # Check points format
        points = zone_data['points']
        if not isinstance(points, list) or len(points) < 3:
            return False
            
        for point in points:
            if not isinstance(point, list) or len(point) != 2:
                return False
            if not all(0 <= coord <= 1 for coord in point):
                return False
                
    return True

# Initialize and validate
if not validate_zone_config(ACTIVE_ZONE_CONFIG):
    raise ValueError(f"Invalid zone configuration: {ACTIVE_ZONE_CONFIG}")