import os

# Set environment for Wayland before other imports
os.environ['QT_QPA_PLATFORM'] = 'wayland'
if 'DISPLAY' in os.environ:
    del os.environ['DISPLAY']

# Standard library imports
import argparse
import json
import os
import sys
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from functools import lru_cache
from http.server import HTTPServer, BaseHTTPRequestHandler

# Third-party imports
import cv2
import numpy as np

# Camera-specific imports
from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics, postprocess_nanodet_detection

# Local imports
from metasettings import (
    get_active_zone_config,
    get_log_path,
    get_system_setting,
    set_external_zone_file,
    SYSTEM_CONFIG
)

class Detection:
    """Represents a detected object with bounding box, category, and confidence."""
    
    def __init__(self, coords, category, conf, metadata):
        """
        Create a Detection object with bounding box coordinates, category and confidence.
        
        Args:
            coords: Raw inference coordinates
            category: Object category index
            conf: Detection confidence score
            metadata: Camera metadata for coordinate conversion
        """
        self.category = category
        self.conf = conf
        self.box = app.imx500.convert_inference_coords(coords, metadata, app.picam2)

class CrosswalkZone:
    """Define a zone (crosswalk, vehicle lane, etc.) using polygon coordinates."""
    
    def __init__(self, name, polygon_points):
        """
        Initialize a zone with a name and polygon boundary points.
        
        Args:
            name: Zone identifier (e.g., 'crosswalk', 'north_lane')
            polygon_points: List of (x, y) coordinates defining the zone boundary
        """
        self.name = name
        self.points = np.array(polygon_points, dtype=np.int32)
    
    def contains_point(self, x, y):
        """
        Check if a point (x,y) is inside this zone using OpenCV's pointPolygonTest.
        
        Args:
            x, y: Point coordinates to test
            
        Returns:
            bool: True if point is inside the zone
        """
        return cv2.pointPolygonTest(self.points, (x, y), False) >= 0
    
    def contains_detection(self, detection):
        """
        Check if detection center is inside this zone.
        
        Args:
            detection: Detection object to test
            
        Returns:
            bool: True if detection center is in this zone
        """
        x, y, w, h = detection.box
        center_x = x + w // 2
        center_y = y + h // 2
        return self.contains_point(center_x, center_y)

class ObjectTracker:
    """Simple object tracking to detect movement across zones."""
    
    def __init__(self, max_disappeared=10):
        """
        Initialize object tracker.
        
        Args:
            max_disappeared: Maximum frames an object can be missing before removal
        """
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        
    def register(self, detection):
        """
        Register a new object for tracking.
        
        Args:
            detection: Detection object to start tracking
        """
        self.objects[self.next_object_id] = {
            'detection': detection,
            'zone_history': [],
            'first_seen': time.time(),
            'last_seen': time.time()
        }
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        
    def deregister(self, object_id):
        """
        Remove an object from tracking.
        
        Args:
            object_id: ID of object to remove from tracking
        """
        del self.objects[object_id]
        del self.disappeared[object_id]
        
    def update(self, detections, zones):
        """
        Update tracked objects with new detections and detect zone crossings.
        
        Args:
            detections: List of current frame detections
            zones: Dictionary of zone objects for boundary checking
        """
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
                        app.crosswalk_monitor.log_crossing_event(
                            object_type=get_labels(app)[int(detection.category)],
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
    """Main crosswalk monitoring system for detecting and logging crossing events."""
    
    def __init__(self):
        """Initialize the crosswalk monitoring system."""
        self.zones = {}
        max_disappeared = get_system_setting('max_disappeared_frames', 10)
        self.tracker = ObjectTracker(max_disappeared)
        max_events = get_system_setting('max_crossing_events', 1000)
        self.crossing_events = deque(maxlen=max_events)
        self.stats = defaultdict(int)
        self.log_path = None
        
    def setup_zones_from_metasettings(self, frame_width, frame_height):
        """
        Load zone definitions from metasettings and convert to absolute coordinates.
        
        Args:
            frame_width: Camera frame width for coordinate conversion
            frame_height: Camera frame height for coordinate conversion
        """
        zone_config = get_active_zone_config()
        zone_data = zone_config['zones']

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
        """
        Log a zone crossing event to file and update statistics.
        
        Args:
            object_type: Type of object that crossed (e.g., 'person', 'car')
            zone_from: Zone the object moved from
            zone_to: Zone the object moved to
            confidence: Detection confidence score
        """
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
        """
        Analyze current detections for crosswalk activity and violations.
        
        Args:
            detections: List of Detection objects from current frame
        """
        labels = get_labels(app)
        
        for detection in detections:
            object_type = labels[int(detection.category)]
            
            # Only track relevant objects
            tracked_objects = get_system_setting('tracked_objects', ['person', 'car', 'truck', 'bus', 'bicycle', 'motorcycle'])
            if object_type in tracked_objects:
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

class CrosswalkDetectionApp:
    """Main application class for crosswalk detection system"""
    
    def __init__(self):
        self.last_detections = []
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.crosswalk_monitor = CrosswalkMonitor()
        self.imx500 = None
        self.intrinsics = None
        self.picam2 = None
        
    def get_latest_frame(self):
        """Thread-safe method to get the latest frame"""
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def update_frame(self, frame):
        """Thread-safe method to update the latest frame"""
        with self.frame_lock:
            self.latest_frame = frame

# Global app instance
app = CrosswalkDetectionApp()

class WebServer:
    """Web server for streaming camera feed with overlay"""
    
    def __init__(self, app_instance):
        self.app = app_instance
        
    def start(self, port=8080):
        """Start the web server in a separate thread"""
        def run_server():
            try:
                server = HTTPServer(('0.0.0.0', port), lambda *args: StreamingHandler(self.app, *args))
                print(f"Web preview available at http://raspberry_pi_ip:{port}")
                print(f"Local access: http://localhost:{port}")
                server.serve_forever()
            except Exception as e:
                print(f"Web server error: {e}")
        
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        return thread

class StreamingHandler(BaseHTTPRequestHandler):
    """HTTP handler for streaming video and serving web interface"""
    
    def __init__(self, app_instance, *args, **kwargs):
        self.app = app_instance
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        if self.path == '/':
            self._serve_html()
        elif self.path == '/stream.mjpg':
            self._serve_video_stream()
    
    def _serve_html(self):
        """Serve the main HTML interface"""
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
                <h1>Crosswalk Detection Monitor</h1>
                <div class="stats">
                    <div class="status">LIVE FEED ACTIVE</div>
                    <p>Detection Threshold: 0.55 | Resolution: 1280x720</p>
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
    
    def _serve_video_stream(self):
        """Serve the MJPEG video stream"""
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
        self.end_headers()
        
        while True:
            try:
                frame = self.app.get_latest_frame()
                if frame is not None:
                    quality = get_system_setting('jpeg_quality', 85)
                    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                    frame_data = jpeg.tobytes()
                else:
                    # Create a placeholder frame
                    placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
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
                
                fps = get_system_setting('web_stream_fps', 10)
                time.sleep(1.0 / fps)  # Dynamic FPS for web stream
                
            except Exception as e:
                print(f"Streaming error: {e}")
                break

def parse_detections(metadata: dict, app_instance, args):
    """Parse the output tensor into a number of detected objects, scaled to the ISP output."""
    bbox_normalization = app_instance.intrinsics.bbox_normalization
    bbox_order = app_instance.intrinsics.bbox_order
    threshold = args.threshold
    iou = args.iou
    max_detections = args.max_detections

    np_outputs = app_instance.imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = app_instance.imx500.get_input_size()
    if np_outputs is None:
        return app_instance.last_detections
    if app_instance.intrinsics.postprocess == "nanodet":
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

    app_instance.last_detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score > threshold
    ]
    
    # Analyze detections for crosswalk activity
    app_instance.crosswalk_monitor.analyze_detections(app_instance.last_detections)
    
    return app_instance.last_detections

def get_labels(app_instance):
    """Get labels from the network intrinsics"""
    labels = app_instance.intrinsics.labels
    if app_instance.intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels

def draw_detections_on_array(frame, detections, zones):
    """
    Draw detection bounding boxes and zone boundaries on video frame for visualization.
    
    Args:
        frame: OpenCV frame array to draw on
        detections: List of Detection objects to visualize
        zones: Dictionary of CrosswalkZone objects to draw
        
    Returns:
        Modified frame with overlay graphics
    """
    if detections is None:
        return frame
    
    labels = get_labels(app)
    
    # Draw crosswalk zones first
    zone_colors = get_system_setting('zone_colors', {})
    for zone_name, zone in zones.items():
        color = zone_colors.get(zone_name, zone_colors.get('default', (128, 128, 128)))
        
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
    detection_colors = get_system_setting('detection_colors', {})
    for detection in detections:
        x, y, w, h = detection.box
        object_type = labels[int(detection.category)]
        label = f"{object_type} ({detection.conf:.2f})"

        # Color code by object type
        color = detection_colors.get(object_type, detection_colors.get('default', (255, 255, 255)))

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
    parser.add_argument("--threshold", type=float, help="Detection threshold (overrides metasettings)")
    parser.add_argument("--iou", type=float, help="Set iou threshold (overrides metasettings)")
    parser.add_argument("--max-detections", type=int, help="Set max detections (overrides metasettings)")
    parser.add_argument("--ignore-dash-labels", action=argparse.BooleanOptionalAction, help="Remove '-' labels ")
    parser.add_argument("--postprocess", choices=["", "nanodet"],
                        default=None, help="Run post process of type")
    parser.add_argument("-r", "--preserve-aspect-ratio", action=argparse.BooleanOptionalAction,
                        help="preserve the pixel aspect ratio of the input tensor")
    parser.add_argument("--labels", type=str, help="Path to the labels file")
    parser.add_argument("--print-intrinsics", action="store_true",
                        help="Print JSON network_intrinsics then exit")
    parser.add_argument("--web-port", type=int, help="Web server port (overrides metasettings)")
    parser.add_argument("--zone-config", type=str, help="Zone configuration file to load (e.g., 'chapel-crosswalk')")
    return parser.parse_args()

def main():
    """Main function to run the crosswalk detection system"""
    args = get_args()

    # Set external zone file if specified
    if args.zone_config:
        set_external_zone_file(args.zone_config)
        print(f"Using external zone configuration: {args.zone_config}")

    # Get settings from metasettings
    imx_model = get_system_setting('imx_model_path')
    log_path = get_log_path()
    app.crosswalk_monitor.log_path = str(log_path)
    
    # Get active zone configuration info
    zone_config = get_active_zone_config()
    print(f"Using zone configuration: {zone_config['name']}")
    print(f"Description: {zone_config['description']}")

    # This must be called before instantiation of Picamera2
    app.imx500 = IMX500(imx_model)
    app.intrinsics = app.imx500.network_intrinsics
    if not app.intrinsics:
        app.intrinsics = NetworkIntrinsics()
        app.intrinsics.task = "object detection"
    elif app.intrinsics.task != "object detection":
        print("Network is not an object detection task", file=sys.stderr)
        exit()

    # Override intrinsics from args
    for key, value in vars(args).items():
        if key == 'labels' and value is not None:
            with open(value, 'r') as f:
                app.intrinsics.labels = f.read().splitlines()
        elif hasattr(app.intrinsics, key) and value is not None:
            setattr(app.intrinsics, key, value)

    # Defaults from metasettings
    if app.intrinsics.labels is None:
        labels_file = get_system_setting('labels_file')
        with open(labels_file, "r") as f:
            app.intrinsics.labels = f.read().splitlines()
    app.intrinsics.update_with_defaults()
    
    # Apply metasettings defaults if not overridden by args
    if args.threshold is None:
        args.threshold = get_system_setting('detection_threshold', 0.55)
    if args.iou is None:
        args.iou = get_system_setting('iou_threshold', 0.65)
    if args.max_detections is None:
        args.max_detections = get_system_setting('max_detections', 10)

    if args.print_intrinsics:
        print(app.intrinsics)
        exit()

    app.picam2 = Picamera2(app.imx500.camera_num)
    config = app.picam2.create_preview_configuration(main={"size": (1280, 720)})
    if app.intrinsics.inference_rate:
        config["controls"] = {"FrameRate": app.intrinsics.inference_rate}
        
    # Setup crosswalk zones based on camera resolution
    frame_width = config['main']['size'][0]
    frame_height = config['main']['size'][1]
    app.crosswalk_monitor.setup_zones_from_metasettings(frame_width, frame_height)
    
    print(f"Crosswalk Monitor Started!")
    print(f"Camera Resolution: {frame_width}x{frame_height}")
    print(f"Detection Threshold: {args.threshold}")
    print(f"Logs will be saved to: {log_path}")
    print("="*50)

    # Start web server
    web_port = args.web_port if args.web_port else get_system_setting('web_port', 8080)
    web_server = WebServer(app)
    web_server.start(web_port)

    app.imx500.show_network_fw_progress_bar()
    
    print("Starting camera...")
    app.picam2.start(config, show_preview=False)  # No Qt preview needed
    print("Camera started successfully!")

    if app.intrinsics.preserve_aspect_ratio:
        app.imx500.set_auto_aspect_ratio()
    
    try:
        while True:
            metadata = app.picam2.capture_metadata()
            last_results = parse_detections(metadata, app, args)

            # Filter only relevant types
            filtered = []
            labels = get_labels(app)
            tracked_objects = get_system_setting('tracked_objects', ['person', 'car', 'truck', 'bus', 'bicycle', 'motorcycle'])
            for det in last_results:
                object_type = labels[int(det.category)]
                if object_type in tracked_objects:
                    filtered.append(det)

            app.crosswalk_monitor.tracker.update(filtered, app.crosswalk_monitor.zones)
            
            # Update web frame
            try:
                frame = app.picam2.capture_array()
                
                # Fix color inversion (RGB to BGR conversion)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                frame_with_overlay = draw_detections_on_array(frame, last_results, app.crosswalk_monitor.zones)
                
                # Update the frame for web streaming
                app.update_frame(frame_with_overlay)
                    
            except Exception as e:
                print(f"Frame processing error: {e}")
                
    except KeyboardInterrupt:
        print("\nCrosswalk Monitor Stopped")
        print(f"Total Events Logged: {len(app.crosswalk_monitor.crossing_events)}")
        print(f"Logs saved to: {log_path}")
        app.picam2.stop()

if __name__ == "__main__":
    main()
