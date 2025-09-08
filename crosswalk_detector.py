import os
print("=== ENVIRONMENT DEBUG ===")
print(f"QT_QPA_PLATFORM: {os.environ.get('QT_QPA_PLATFORM', 'NOT SET')}")
print(f"DISPLAY: {os.environ.get('DISPLAY', 'NOT SET')}")
print(f"XDG_SESSION_TYPE: {os.environ.get('XDG_SESSION_TYPE', 'NOT SET')}")
print(f"WAYLAND_DISPLAY: {os.environ.get('WAYLAND_DISPLAY', 'NOT SET')}")
print("========================")

# Set environment BEFORE any other imports
os.environ['QT_QPA_PLATFORM'] = 'wayland'
# Make sure DISPLAY is not set when using Wayland
if 'DISPLAY' in os.environ:
    del os.environ['DISPLAY']

print(f"After setting - QT_QPA_PLATFORM: {os.environ.get('QT_QPA_PLATFORM')}")

import argparse
import sys
import json
import time
import threading
from datetime import datetime
from functools import lru_cache
from collections import defaultdict, deque
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver

import cv2
import numpy as np

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics,
                                      postprocess_nanodet_detection)

last_detections = []
latest_frame = None
frame_lock = threading.Lock()

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
        
    def update(self, detections, zones):
        """Update tracked objects with new detections"""
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return

        if len(self.objects) == 0:
            for detection in detections:
                self.register(detection)
        else:
            for detection in detections:
                # Very simple matching to one object for now
                if self.objects:
                    object_id = list(self.objects.keys())[0]
                    obj = self.objects[object_id]

                    prev_zones = obj['zone_history'][-1] if obj['zone_history'] else None

                    # Determine current zone(s) for this detection
                    current_zones = []
                    for zone_name, zone in zones.items():
                        if zone.contains_detection(detection):
                            current_zones.append(zone_name)

                    # Pick just one (or extend to support multiple later)
                    current_zone = current_zones[0] if current_zones else None

                    # Check for zone transition
                    if prev_zones and current_zone and prev_zones != current_zone:
                        crosswalk_monitor.log_crossing_event(
                            object_type=get_labels()[int(detection.category)],
                            zone_from=prev_zones,
                            zone_to=current_zone,
                            confidence=detection.conf
                        )

                    # Update the object
                    obj['detection'] = detection
                    obj['last_seen'] = time.time()
                    self.disappeared[object_id] = 0
                    if current_zone:
                        obj['zone_history'].append(current_zone)

class CrosswalkMonitor:
    """Main crosswalk monitoring system"""
    def __init__(self):
        self.zones = {}
        self.tracker = ObjectTracker()
        self.crossing_events = deque(maxlen=1000)  # Keep last 1000 events
        self.stats = defaultdict(int)
        self.log_path = None
        
    def setup_zones_from_file(self, json_path, frame_width, frame_height):
        """Setup zone boundaries with json"""
        with open(json_path, "r") as f:
            zone_data = json.load(f)

        self.zones = {}
        for key, value in zone_data.items():
            name = value["name"]
            norm_points = value["points"]
            abs_points = [
                (int(x * frame_width), int(y * frame_height))
                for (x, y) in norm_points
            ]
            self.zones[key] = CrosswalkZone(name, abs_points)
        
    def log_crossing_event(self, object_type, zone_from, zone_to, confidence):
        """Log a crossing event"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'object_type': object_type,
            'from_zone': zone_from,
            'to_zone': zone_to,
            'confidence': float(confidence)
        }
        self.crossing_events.append(event)
        self.stats[f'{object_type}_crossings'] += 1
        
        # Print to console for real-time monitoring
        print(f"CROSSING DETECTED: {object_type} from {zone_from} to {zone_to} (conf: {confidence:.2f})")
        
        with open(self.log_path, 'a') as f:
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

# Web Server for Preview
class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Crosswalk Monitor</title>
                <style>
                    body { font-family: Arial, sans-serif; background: #1a1a1a; color: white; margin: 0; padding: 20px; }
                    .container { max-width: 1200px; margin: 0 auto; }
                    h1 { color: #ffcc00; text-align: center; }
                    .video-container { text-align: center; margin: 20px 0; }
                    .stats { background: #333; padding: 15px; border-radius: 8px; margin: 20px 0; }
                    .status { color: #00ff00; font-weight: bold; }
                    img { border: 2px solid #555; border-radius: 8px; max-width: 100%; }
                    .controls { text-align: center; margin: 10px 0; }
                    button { background: #ffcc00; color: black; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
                    button:hover { background: #ffd700; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🚦 Crosswalk Detection Monitor</h1>
                    <div class="stats">
                        <div class="status">● LIVE FEED ACTIVE</div>
                        <p>Detection Threshold: 0.55 | Resolution: 640x480</p>
                    </div>
                    <div class="video-container">
                        <img id="stream" src="/stream.mjpg" alt="Crosswalk Monitor Feed">
                    </div>
                    <div class="controls">
                        <button onclick="location.reload()">Refresh Feed</button>
                        <button onclick="toggleFullscreen()">Toggle Fullscreen</button>
                    </div>
                </div>
                <script>
                    function toggleFullscreen() {
                        const img = document.getElementById('stream');
                        if (img.requestFullscreen) {
                            img.requestFullscreen();
                        }
                    }
                    // Auto-refresh if stream fails
                    document.getElementById('stream').onerror = function() {
                        setTimeout(() => location.reload(), 2000);
                    };
                </script>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
            
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            
            while True:
                try:
                    with frame_lock:
                        if latest_frame is not None:
                            _, jpeg = cv2.imencode('.jpg', latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            frame_data = jpeg.tobytes()
                        else:
                            # Create a placeholder frame
                            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                            cv2.putText(placeholder, 'Starting Camera...', (150, 240), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            _, jpeg = cv2.imencode('.jpg', placeholder)
                            frame_data = jpeg.tobytes()
                    
                    self.wfile.write(b'--jpgboundary\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(frame_data)))
                    self.end_headers()
                    self.wfile.write(frame_data)
                    self.wfile.write(b'\r\n')
                    
                    time.sleep(0.1)  # ~10 FPS for web stream
                    
                except Exception as e:
                    print(f"Streaming error: {e}")
                    break

def start_web_server(port=8080):
    """Start the web server in a separate thread"""
    def run_server():
        try:
            server = HTTPServer(('0.0.0.0', port), StreamingHandler)
            print(f"🌐 Web preview available at http://raspberry_pi_ip:{port}")
            print(f"🌐 Local access: http://localhost:{port}")
            server.serve_forever()
        except Exception as e:
            print(f"Web server error: {e}")
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    return thread

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

def draw_detections_on_array(frame, detections, zones):
    """Draw detections and zones directly on a numpy array for web display"""
    if detections is None:
        return frame
    
    labels = get_labels()
    
    # Draw crosswalk zones first
    for zone_name, zone in zones.items():
        color = {
            'crosswalk': (0, 255, 255),      # Yellow for crosswalk
            'north_lane': (255, 0, 0),       # Blue for north lane  
            'south_lane': (0, 0, 255),       # Red for south lane
            'east_lane': (0, 255, 0),        # Green for east lane
            'west_lane': (150, 75, 0),       # Brown for west lane        
        }.get(zone_name, (128, 128, 128))    # Gray for others
        
        # Draw zone boundary
        cv2.polylines(frame, [zone.points], True, color, 2)
        
        # Draw corner points
        for point in zone.points:
            cv2.circle(frame, tuple(point), radius=5, color=color, thickness=-1)
        
        # Label the zone
        cv2.putText(frame, zone_name.upper(), 
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

        # Draw detection box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness=2)
        
        # Draw center point
        center_x = x + w // 2
        center_y = y + h // 2
        cv2.circle(frame, (center_x, center_y), radius=3, color=color, thickness=-1)
        
        # Draw label with background
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_x = x + 5
        text_y = y + 15
        
        # Background rectangle
        cv2.rectangle(frame, 
                     (text_x, text_y - text_height), 
                     (text_x + text_width, text_y + baseline),
                     (255, 255, 255), cv2.FILLED)
        
        # Text
        cv2.putText(frame, label, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return frame

def get_args():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--labels", type=str, help="Path to the labels file")
    parser.add_argument("--print-intrinsics", action="store_true",
                        help="Print JSON network_intrinsics then exit")
    parser.add_argument("--zone-config", type=str, default="zones/test.json",
                        help="Path to zone definition JSON")
    parser.add_argument("--web-port", type=int, default=8080, help="Web server port")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    imx_model = r"/usr/share/imx500-models/imx500_network_nanodet_plus_416x416_pp.rpk"

    zone_config_filename = os.path.splitext(os.path.basename(args.zone_config))[0]
    log_path = os.path.join("logs", f"{zone_config_filename}.jsonl")
    crosswalk_monitor.log_path = log_path

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
    config = picam2.create_preview_configuration()
    if intrinsics.inference_rate:
        config["controls"] = {"FrameRate": intrinsics.inference_rate}
        
    # Setup crosswalk zones based on camera resolution
    frame_width = config['main']['size'][0]
    frame_height = config['main']['size'][1]
    crosswalk_monitor.setup_zones_from_file(args.zone_config, frame_width, frame_height)
    
    print(f"🚦 Crosswalk Monitor Started!")
    print(f"📹 Camera Resolution: {frame_width}x{frame_height}")
    print(f"🎯 Detection Threshold: {args.threshold}")
    print(f"📝 Logs will be saved to: {log_path}")
    print("="*50)

    # Start web server
    start_web_server(args.web_port)

    imx500.show_network_fw_progress_bar()
    
    print("Starting camera...")
    picam2.start(config, show_preview=False)  # No Qt preview needed
    print("Camera started successfully!")

    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()

    last_results = None
    
    try:
        while True:
            metadata = picam2.capture_metadata()
            last_results = parse_detections(metadata)

            # Filter only relevant types
            filtered = []
            labels = get_labels()
            for det in last_results:
                object_type = labels[int(det.category)]
                if object_type in ['person', 'car', 'truck', 'bus', 'bicycle', 'motorcycle']:
                    filtered.append(det)

            crosswalk_monitor.tracker.update(filtered, crosswalk_monitor.zones)
            
            # Update web frame
            try:
                frame = picam2.capture_array()
                frame_with_overlay = draw_detections_on_array(frame, last_results, crosswalk_monitor.zones)
                
                # Update the global frame for web streaming
                with frame_lock:
                    latest_frame = frame_with_overlay.copy()
                    
            except Exception as e:
                print(f"Frame processing error: {e}")
                
    except KeyboardInterrupt:
        print("\n🛑 Crosswalk Monitor Stopped")
        print(f"📊 Total Events Logged: {len(crosswalk_monitor.crossing_events)}")
        print(f"📝 Logs saved to: {log_path}")
        picam2.stop()
