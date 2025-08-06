import argparse
import sys
import json
import time
from datetime import datetime
from functools import lru_cache
from collections import defaultdict, deque

import cv2
import numpy as np

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics,
                                      postprocess_nanodet_detection)

last_detections = []


class Detection:
    def __init__(self, coords, category, conf, metadata):
        """Create a Detection object, recording the bounding box, category and confidence."""
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)


class CrosswalkZone:
    """Define a zone (crosswalk, vehicle lane, etc.) using polygon coordinates"""
    def __init__(self, name, polygon_points):
        self.name = name
        self.points = np.array(polygon_points, dtype=np.int32)
    
    def contains_point(self, x, y):
        """Check if a point (x,y) is inside this zone using OpenCV's pointPolygonTest"""
        return cv2.pointPolygonTest(self.points, (x, y), False) >= 0
    
    def contains_detection(self, detection):
        """Check if detection center is in this zone"""
        x, y, w, h = detection.box
        center_x = x + w // 2
        center_y = y + h // 2
        return self.contains_point(center_x, center_y)


class ObjectTracker:
    """Simple object tracking to detect movement across zones"""
    def __init__(self, max_disappeared=10):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        
    def register(self, detection):
        """Register a new object"""
        self.objects[self.next_object_id] = {
            'detection': detection,
            'zone_history': [],
            'first_seen': time.time(),
            'last_seen': time.time()
        }
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        
    def deregister(self, object_id):
        """Remove an object from tracking"""
        del self.objects[object_id]
        del self.disappeared[object_id]
        
    def update(self, detections):
        """Update tracked objects with new detections"""
        if len(detections) == 0:
            # Mark all existing objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return
        
        # Simple nearest-neighbor tracking (for production, use more sophisticated methods)
        if len(self.objects) == 0:
            for detection in detections:
                self.register(detection)
        else:
            # Match detections to existing objects (simplified)
            for detection in detections:
                # For now, just update the first object (this is very basic)
                # In production, you'd use distance/IoU matching
                if self.objects:
                    object_id = list(self.objects.keys())[0]
                    self.objects[object_id]['detection'] = detection
                    self.objects[object_id]['last_seen'] = time.time()
                    self.disappeared[object_id] = 0


class CrosswalkMonitor:
    """Main crosswalk monitoring system"""
    def __init__(self):
        self.zones = {}
        self.tracker = ObjectTracker()
        self.crossing_events = deque(maxlen=1000)  # Keep last 1000 events
        self.stats = defaultdict(int)
        
    def setup_zones(self, frame_width, frame_height):
        """Setup crosswalk zones - customize these coordinates for your specific crosswalk"""
        # Example zones - YOU NEED TO ADJUST THESE FOR YOUR ACTUAL CROSSWALK
        self.zones = {
            'crosswalk': CrosswalkZone('pedestrian_crossing', [
                (frame_width * 0.2, frame_height * 0.4),
                (frame_width * 0.8, frame_height * 0.4),
                (frame_width * 0.8, frame_height * 0.6),
                (frame_width * 0.2, frame_height * 0.6)
            ]),
            'north_lane': CrosswalkZone('vehicle_lane_north', [
                (frame_width * 0.1, frame_height * 0.1),
                (frame_width * 0.9, frame_height * 0.1),
                (frame_width * 0.9, frame_height * 0.35),
                (frame_width * 0.1, frame_height * 0.35)
            ]),
            'south_lane': CrosswalkZone('vehicle_lane_south', [
                (frame_width * 0.1, frame_height * 0.65),
                (frame_width * 0.9, frame_height * 0.65),
                (frame_width * 0.9, frame_height * 0.9),
                (frame_width * 0.1, frame_height * 0.9)
            ])
        }
        
    def log_crossing_event(self, object_type, zone_from, zone_to, confidence):
        """Log a crossing event"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'object_type': object_type,
            'from_zone': zone_from,
            'to_zone': zone_to,
            'confidence': confidence
        }
        self.crossing_events.append(event)
        self.stats[f'{object_type}_crossings'] += 1
        
        # Print to console for real-time monitoring
        print(f"CROSSING DETECTED: {object_type} from {zone_from} to {zone_to} (conf: {confidence:.2f})")
        
        # Save to file (optional)
        with open('logs/crossings.jsonl', 'a') as f:
            f.write(json.dumps(event) + '\n')
    
    def analyze_detections(self, detections):
        """Analyze detections for crosswalk activity"""
        labels = get_labels()
        
        for detection in detections:
            object_type = labels[int(detection.category)]
            
            # Only track relevant objects
            if object_type in ['person', 'car', 'truck', 'bus', 'bicycle', 'motorcycle']:
                # Check which zones this detection is in
                current_zones = []
                for zone_name, zone in self.zones.items():
                    if zone.contains_detection(detection):
                        current_zones.append(zone_name)
                
                # Simple crossing detection (in production, use the tracker)
                if current_zones:
                    zone_str = '+'.join(current_zones)
                    print(f"{object_type} in {zone_str} (conf: {detection.conf:.2f})")
                    
                    # Example: detect pedestrian in crosswalk
                    if 'crosswalk' in current_zones and object_type == 'person':
                        print(f"PEDESTRIAN IN CROSSWALK! (confidence: {detection.conf:.2f})")
                    
                    # Example: detect vehicle in crosswalk (potential violation)
                    if 'crosswalk' in current_zones and object_type in ['car', 'truck', 'bus']:
                        print(f"VEHICLE IN CROSSWALK! (confidence: {detection.conf:.2f})")


# Initialize crosswalk monitor
crosswalk_monitor = CrosswalkMonitor()


def parse_detections(metadata: dict):
    """Parse the output tensor into a number of detected objects, scaled to the ISP output."""
    global last_detections
    bbox_normalization = intrinsics.bbox_normalization
    bbox_order = intrinsics.bbox_order
    threshold = args.threshold
    iou = args.iou
    max_detections = args.max_detections

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()
    if np_outputs is None:
        return last_detections
    if intrinsics.postprocess == "nanodet":
        boxes, scores, classes = \
            postprocess_nanodet_detection(outputs=np_outputs[0], conf=threshold, iou_thres=iou,
                                          max_out_dets=max_detections)[0]
        from picamera2.devices.imx500.postprocess import scale_boxes
        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
    else:
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if bbox_normalization:
            boxes = boxes / input_h

        if bbox_order == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]
        boxes = np.array_split(boxes, 4, axis=1)
        boxes = zip(*boxes)

    last_detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score > threshold
    ]
    
    # Analyze detections for crosswalk activity
    crosswalk_monitor.analyze_detections(last_detections)
    
    return last_detections


@lru_cache
def get_labels():
    labels = intrinsics.labels

    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels


def draw_detections(request, stream="main"):
    """Draw the detections and zones onto the ISP output."""
    detections = last_results
    if detections is None:
        return
    labels = get_labels()
    with MappedArray(request, stream) as m:
        # Draw crosswalk zones first
        for zone_name, zone in crosswalk_monitor.zones.items():
            color = {
                'crosswalk': (0, 255, 255),      # Yellow for crosswalk
                'north_lane': (255, 0, 0),       # Blue for north lane  
                'south_lane': (0, 0, 255)        # Red for south lane
            }.get(zone_name, (128, 128, 128))    # Gray for others
            
            # Draw zone boundary
            cv2.polylines(m.array, [zone.points], True, color, 2)
            
            for point in zone.points:
                cv2.circle(m.array, tuple(point), radius=5, color=color, thickness=-1)
            
            # Label the zone
            cv2.putText(m.array, zone_name.upper(), 
                       tuple(zone.points[0]), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, color, 2)
        
        # Draw detections
        for detection in detections:
            x, y, w, h = detection.box
            object_type = labels[int(detection.category)]
            label = f"{object_type} ({detection.conf:.2f})"

            # Color code by object type
            color = {
                'person': (0, 255, 0),      # Green for people
                'car': (255, 0, 0),         # Blue for cars
                'truck': (255, 0, 0),       # Blue for trucks
                'bus': (255, 0, 0),         # Blue for buses
                'bicycle': (0, 255, 255),   # Yellow for bikes
                'motorcycle': (0, 255, 255) # Yellow for motorcycles
            }.get(object_type, (255, 255, 255))  # White for others

            # Calculate text size and position
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = x + 5
            text_y = y + 15

            # Create a copy of the array to draw the background with opacity
            overlay = m.array.copy()

            # Draw the background rectangle on the overlay
            cv2.rectangle(overlay,
                          (text_x, text_y - text_height),
                          (text_x + text_width, text_y + baseline),
                          (255, 255, 255),  # Background color (white)
                          cv2.FILLED)

            alpha = 0.30
            cv2.addWeighted(overlay, alpha, m.array, 1 - alpha, 0, m.array)

            # Draw text on top of the background
            cv2.putText(m.array, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw detection box
            cv2.rectangle(m.array, (x, y), (x + w, y + h), color, thickness=2)

        if intrinsics.preserve_aspect_ratio:
            b_x, b_y, b_w, b_h = imx500.get_roi_scaled(request)
            color = (255, 0, 0)  # red
            cv2.putText(m.array, "ROI", (b_x + 5, b_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.rectangle(m.array, (b_x, b_y), (b_x + b_w, b_y + b_h), (255, 0, 0, 0))


def get_args():
    parser = argparse.ArgumentParser()
#     parser.add_argument("--model", type=str, help="Path of the model",
#                         default="/usr/share/imx500-models/imx500_network_nanodet_plus_416x416_pp.rpk")
    parser.add_argument("--fps", type=int, help="Frames per second")
    parser.add_argument("--bbox-normalization", action=argparse.BooleanOptionalAction, help="Normalize bbox")
    parser.add_argument("--bbox-order", choices=["yx", "xy"], default="yx",
                        help="Set bbox order yx -> (y0, x0, y1, x1) xy -> (x0, y0, x1, y1)")
    parser.add_argument("--threshold", type=float, default=0.55, help="Detection threshold")
    parser.add_argument("--iou", type=float, default=0.65, help="Set iou threshold")
    parser.add_argument("--max-detections", type=int, default=10, help="Set max detections")
    parser.add_argument("--ignore-dash-labels", action=argparse.BooleanOptionalAction, help="Remove '-' labels ")
    parser.add_argument("--postprocess", choices=["", "nanodet"],
                        default=None, help="Run post process of type")
    parser.add_argument("-r", "--preserve-aspect-ratio", action=argparse.BooleanOptionalAction,
                        help="preserve the pixel aspect ratio of the input tensor")
    parser.add_argument("--labels", type=str,
                        help="Path to the labels file")
    parser.add_argument("--print-intrinsics", action="store_true",
                        help="Print JSON network_intrinsics then exit")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    imx_model = r"/usr/share/imx500-models/imx500_network_nanodet_plus_416x416_pp.rpk"

    # This must be called before instantiation of Picamera2
    imx500 = IMX500(imx_model)
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
    elif intrinsics.task != "object detection":
        print("Network is not an object detection task", file=sys.stderr)
        exit()

    # Override intrinsics from args
    for key, value in vars(args).items():
        if key == 'labels' and value is not None:
            with open(value, 'r') as f:
                intrinsics.labels = f.read().splitlines()
        elif hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)

    # Defaults
    if intrinsics.labels is None:
        with open("assets/coco_labels.txt", "r") as f:
            intrinsics.labels = f.read().splitlines()
    intrinsics.update_with_defaults()

    if args.print_intrinsics:
        print(intrinsics)
        exit()

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12)

    # Setup crosswalk zones based on camera resolution
    frame_width = config['main']['size'][0]
    frame_height = config['main']['size'][1]
    crosswalk_monitor.setup_zones(frame_width, frame_height)
    
    print(f"Crosswalk Monitor Started!")
    print(f"Camera Resolution: {frame_width}x{frame_height}")
    print(f"Detection Threshold: {args.threshold}")
    print(f"Logs will be saved to: logs/crossings.jsonl")
    print("="*50)

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=True)

    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()

    last_results = None
    picam2.pre_callback = draw_detections
    
    try:
        while True:
            last_results = parse_detections(picam2.capture_metadata())
    except KeyboardInterrupt:
        print("\nCrosswalk Monitor Stopped")
        print(f"Total Events Logged: {len(crosswalk_monitor.crossing_events)}")
        print("Check logs/crossings.jsonl for detailed logs")
        picam2.stop()