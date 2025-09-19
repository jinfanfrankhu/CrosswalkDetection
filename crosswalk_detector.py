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
    get_detection_threshold,
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
        Uses weighted center point for better vehicle detection.

        Args:
            detection: Detection object to test

        Returns:
            bool: True if detection center is in this zone
        """
        x, y, w, h = detection.box
        center_x = x + w // 2

        # Weight the center point lower for better vehicle detection
        # For vehicles, the important part is closer to the bottom of the bounding box
        center_y = y + int(h * 0.75)  # 75% down from top instead of 50%

        return self.contains_point(center_x, center_y)

class TriggerBasedTracker:
    """Simple trigger-based crossing detection that works with unreliable object detection."""

    def __init__(self, max_disappeared=5, max_distance=100):
        """
        Initialize trigger-based tracker.

        Args:
            max_disappeared: Maximum frames an object can be missing before removal
            max_distance: Maximum pixel distance for object matching
        """
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.recent_triggers = {}  # Track recent triggers to avoid duplicates

    def register(self, detection):
        """
        Register a new object for tracking.

        Args:
            detection: Detection object to start tracking
        """
        center_x, center_y = self._get_detection_center(detection)

        self.objects[self.next_object_id] = {
            'detection': detection,
            'current_zone': None,
            'last_zone': None,
            'position': (center_x, center_y),
            'position_history': [(center_x, center_y)],  # Add position history
            'first_seen': time.time(),
            'last_seen': time.time(),
            'object_type': None
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

    def _calculate_distance(self, center1, center2):
        """Calculate Euclidean distance between two points."""
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    def _get_detection_center(self, detection):
        """
        Get weighted center point of a detection.
        Uses lower weighting for better vehicle tracking.
        """
        x, y, w, h = detection.box
        center_x = x + w // 2
        # Weight center point lower (75% down) for better vehicle detection
        center_y = y + int(h * 0.75)
        return (center_x, center_y)

    def _match_detections_to_objects(self, detections):
        """
        Match current detections to existing tracked objects.
        Returns list of (object_id, detection) pairs and unmatched detections.
        """
        if not self.objects:
            return [], detections

        matches = []
        unmatched_detections = []

        for detection in detections:
            detection_center = self._get_detection_center(detection)
            best_match = None
            min_distance = float('inf')

            for object_id, obj in self.objects.items():
                if obj['position_history']:
                    last_position = obj['position_history'][-1]
                    distance = self._calculate_distance(detection_center, last_position)

                    if distance < min_distance and distance < self.max_distance:
                        min_distance = distance
                        best_match = object_id

            if best_match is not None:
                matches.append((best_match, detection))
            else:
                unmatched_detections.append(detection)

        return matches, unmatched_detections

    def _should_trigger_crossing(self, object_type, from_zone, to_zone):
        """
        Determine if a zone transition should trigger a crossing event.

        Args:
            object_type: Type of detected object
            from_zone: Zone object was moving from
            to_zone: Zone object is moving to

        Returns:
            bool: True if this transition should log a crossing
        """
        if not from_zone or not to_zone:
            return False

        # Vehicle crossing rules: main_road → crosswalk = immediate violation/crossing
        if object_type in ['car', 'truck', 'bus', 'motorcycle']:
            if from_zone == 'main_road' and to_zone == 'crosswalk':
                return True

        # Pedestrian crossing rules: any → crosswalk = crossing attempt
        elif object_type in ['person', 'pedestrian']:
            if to_zone == 'crosswalk' and from_zone != 'crosswalk':
                return True

        return False

    def _get_trigger_key(self, object_type, from_zone, to_zone):
        """Generate a key for tracking recent triggers to avoid duplicates."""
        return f"{object_type}_{from_zone}_{to_zone}"

    def _can_trigger(self, trigger_key, obj, cooldown_seconds=10):
        """
        Check if enough time has passed since last trigger of this type.
        Implements distance-based reset for vehicles approaching from earlier positions.

        Args:
            trigger_key: Unique key for this trigger type
            obj: Object data containing position and type information
            cooldown_seconds: Minimum seconds between same triggers

        Returns:
            bool: True if trigger is allowed
        """
        current_time = time.time()
        current_position = obj['position']

        if trigger_key in self.recent_triggers:
            last_data = self.recent_triggers[trigger_key]
            time_since_last = current_time - last_data['time']

            # Distance-based reset for vehicles
            if obj['object_type'] in ['car', 'truck', 'bus', 'motorcycle']:
                last_position = last_data.get('position')
                if last_position:
                    # Calculate Euclidean distance between current and last position
                    distance = self._calculate_distance(current_position, last_position)

                    # If vehicle is detected 100+ pixels away from last trigger position,
                    # this likely indicates a different vehicle, so reset cooldown
                    if distance > 100:
                        print(f"🔄 Cooldown reset: {obj['object_type']} detected {distance:.0f}px away from last trigger (likely new vehicle)")
                        self.recent_triggers[trigger_key] = {
                            'time': current_time,
                            'position': current_position
                        }
                        return True

            # Standard cooldown check
            if time_since_last < cooldown_seconds:
                return False

        # Update trigger record with position data
        self.recent_triggers[trigger_key] = {
            'time': current_time,
            'position': current_position
        }
        return True

    def update(self, detections, zones):
        """
        Trigger-based update method that logs crossings immediately on specific zone transitions.

        Args:
            detections: List of current frame detections
            zones: Dictionary of zone objects for boundary checking
        """
        labels = get_labels(app)

        # Handle case where no detections are found
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return

        # Match detections to existing objects
        matches, unmatched_detections = self._match_detections_to_objects(detections)

        # Update matched objects and check for trigger events
        for object_id, detection in matches:
            obj = self.objects[object_id]
            detection_center = self._get_detection_center(detection)

            # Update object info
            obj['detection'] = detection
            obj['last_seen'] = time.time()
            obj['object_type'] = labels[int(detection.category)]
            obj['position'] = detection_center

            # Update position history (keep last 10 positions)
            if 'position_history' not in obj:
                obj['position_history'] = []
            obj['position_history'].append(detection_center)
            if len(obj['position_history']) > 10:
                obj['position_history'] = obj['position_history'][-10:]

            self.disappeared[object_id] = 0

            # Determine current zone for this detection
            current_zone = None
            for zone_name, zone in zones.items():
                if zone.contains_detection(detection):
                    current_zone = zone_name
                    break  # Take first match

            # Check for zone transition and trigger events
            if current_zone != obj['current_zone']:
                from_zone = obj['current_zone']
                to_zone = current_zone

                # Update zone tracking
                obj['last_zone'] = from_zone
                obj['current_zone'] = to_zone

                # Check if this transition should trigger a crossing event
                if self._should_trigger_crossing(obj['object_type'], from_zone, to_zone):
                    trigger_key = self._get_trigger_key(obj['object_type'], from_zone, to_zone)

                    # Only trigger if cooldown period has passed (with distance-based reset)
                    if self._can_trigger(trigger_key, obj):
                        self._log_immediate_crossing(obj, from_zone, to_zone)

        # Register new objects for unmatched detections
        for detection in unmatched_detections:
            self.register(detection)

        # Increment disappeared counter for unmatched existing objects
        matched_object_ids = {object_id for object_id, _ in matches}
        for object_id in self.objects:
            if object_id not in matched_object_ids:
                self.disappeared[object_id] += 1

    def _log_immediate_crossing(self, obj, from_zone, to_zone):
        """Log an immediate crossing trigger event."""
        trigger_type = "vehicle_violation" if obj['object_type'] in ['car', 'truck', 'bus', 'motorcycle'] else "pedestrian_crossing"

        crossing_event = {
            'timestamp': datetime.now().isoformat(),
            'object_type': obj['object_type'],
            'trigger_type': trigger_type,
            'from_zone': from_zone,
            'to_zone': to_zone,
            'confidence': obj['detection'].conf if obj['detection'] else 0.0,
            'detection_time': obj['last_seen']
        }

        # Log the crossing through the monitor
        app.crosswalk_monitor.log_immediate_crossing(crossing_event)

        # Print immediate feedback
        if trigger_type == "vehicle_violation":
            print(f"🚗 VEHICLE CROSSING: {obj['object_type']} from {from_zone} → {to_zone} (conf: {crossing_event['confidence']:.2f})")
        else:
            print(f"🚶 PEDESTRIAN CROSSING: {obj['object_type']} entering crosswalk from {from_zone} (conf: {crossing_event['confidence']:.2f})")

class CrosswalkMonitor:
    """Main crosswalk monitoring system for detecting and logging crossing events."""

    def __init__(self):
        """Initialize the crosswalk monitoring system."""
        self.zones = {}
        max_disappeared = get_system_setting('max_disappeared_frames', 5)  # Shorter for unreliable detection
        max_distance = get_system_setting('max_tracking_distance', 100)
        self.tracker = TriggerBasedTracker(max_disappeared, max_distance)
        max_events = get_system_setting('max_crossing_events', 1000)
        self.crossing_events = deque(maxlen=max_events)
        self.stats = defaultdict(int)
        self.log_path = None

        # Separate log files for different entity types
        self.log_files = {
            'pedestrian': None,  # Will be set in setup
            'vehicle': None,     # Will be set in setup
            'all': None         # Combined log
        }
        
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

    def setup_log_files(self, base_log_path, zone_config_name):
        """
        Setup separate log files for different entity types in organized folders.

        Args:
            base_log_path: Base path for log files
            zone_config_name: Name of the zone configuration for folder organization
        """
        import os
        from pathlib import Path

        base_path = Path(base_log_path)
        base_log_dir = base_path.parent
        base_name = base_path.stem

        # Create zone-specific log directory
        zone_log_dir = base_log_dir / zone_config_name
        zone_log_dir.mkdir(parents=True, exist_ok=True)

        # Create separate log files in the zone directory
        self.log_files['pedestrian'] = zone_log_dir / f"{base_name}_pedestrians.jsonl"
        self.log_files['vehicle'] = zone_log_dir / f"{base_name}_vehicles.jsonl"
        self.log_files['all'] = zone_log_dir / f"{base_name}_all.jsonl"

        # Create log files with headers
        for log_type, log_path in self.log_files.items():
            if log_path and not log_path.exists():
                with open(log_path, 'w') as f:
                    header = {
                        'log_type': log_type,
                        'zone_config': zone_config_name,
                        'created': datetime.now().isoformat(),
                        'description': f'Crosswalk crossing events for {log_type} in {zone_config_name}'
                    }
                    f.write(json.dumps(header, default=self._json_serializer) + '\n')

        print(f"Log files setup for zone '{zone_config_name}':")
        print(f"  Directory: {zone_log_dir}")
        print(f"  Pedestrians: {self.log_files['pedestrian'].name}")
        print(f"  Vehicles: {self.log_files['vehicle'].name}")
        print(f"  All events: {self.log_files['all'].name}")

    def _categorize_object_type(self, object_type):
        """
        Categorize object type into pedestrian, vehicle, or other.

        Args:
            object_type: The detected object type

        Returns:
            Category string: 'pedestrian', 'vehicle', or 'other'
        """
        pedestrian_types = ['person', 'pedestrian']
        vehicle_types = ['car', 'truck', 'bus', 'bicycle', 'motorcycle', 'vehicle']

        if object_type.lower() in pedestrian_types:
            return 'pedestrian'
        elif object_type.lower() in vehicle_types:
            return 'vehicle'
        else:
            return 'other'

    def _json_serializer(self, obj):
        """
        Custom JSON serializer to handle NumPy data types.

        Args:
            obj: Object to serialize

        Returns:
            JSON-serializable version of the object
        """
        import numpy as np

        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()

        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def log_immediate_crossing(self, crossing_event):
        """
        Log an immediate crossing trigger event to appropriate files.

        Args:
            crossing_event: Dictionary containing crossing trigger information
        """
        # Categorize the object
        category = self._categorize_object_type(crossing_event['object_type'])
        crossing_event['category'] = category

        # Update statistics
        self.stats[f'{crossing_event["trigger_type"]}_events'] += 1
        self.stats[f'{category}_triggers'] += 1
        self.stats['total_triggers'] += 1

        # Add trigger to in-memory storage
        self.crossing_events.append(crossing_event)

        # Write to appropriate log files
        log_paths = [self.log_files['all']]  # Always log to main file

        if category in self.log_files:
            log_paths.append(self.log_files[category])

        for log_path in log_paths:
            if log_path:
                try:
                    with open(log_path, 'a') as f:
                        f.write(json.dumps(crossing_event, default=self._json_serializer) + '\n')
                except Exception as e:
                    print(f"Error writing trigger event to {log_path}: {e}")


    def log_crossing_event(self, object_type, zone_from, zone_to, confidence):
        """
        Legacy method - Log a simple zone transition event.

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
            'confidence': float(confidence),
            'event_type': 'zone_transition'
        }
        self.crossing_events.append(event)
        self.stats[f'{object_type}_transitions'] += 1

        # Print to console for real-time monitoring
        print(f"ZONE TRANSITION: {object_type} from {zone_from} to {zone_to} (conf: {confidence:.2f})")

        # Write to main log file only
        if self.log_files['all']:
            try:
                with open(self.log_files['all'], 'a') as f:
                    f.write(json.dumps(event, default=self._json_serializer) + '\n')
            except Exception as e:
                print(f"Error writing transition event: {e}")
    
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
                
                # Legacy detection output (reduced spam for stationary objects)
                if current_zones:
                    # Only print occasional updates for objects in crosswalk to reduce spam
                    import time
                    if not hasattr(self, '_last_zone_print'):
                        self._last_zone_print = {}

                    zone_str = '+'.join(current_zones)
                    current_time = time.time()

                    # Only print every 5 seconds for the same object type in same zone
                    print_key = f"{object_type}_{zone_str}"
                    if (print_key not in self._last_zone_print or
                        current_time - self._last_zone_print[print_key] > 5.0):

                        self._last_zone_print[print_key] = current_time

                        # Only print if it's an interesting zone combination
                        if 'crosswalk' in current_zones:
                            if object_type == 'person':
                                print(f"PEDESTRIAN IN CROSSWALK (confidence: {detection.conf:.2f})")
                            elif object_type in ['car', 'truck', 'bus']:
                                print(f"VEHICLE IN CROSSWALK (confidence: {detection.conf:.2f})")
                        # For debugging: comment out this line to reduce general zone spam
                        # print(f"{object_type} in {zone_str} (conf: {detection.conf:.2f})")

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
        self.yolo_net = None

    def load_yolo_model(self):
        """Load YOLOv8 model for ROI enhancement"""
        try:
            yolo_model_path = get_system_setting('yolo_model_path')
            if not os.path.exists(yolo_model_path):
                print(f"Warning: YOLO model not found at {yolo_model_path}")
                print("ROI enhancement will be disabled")
                return False

            self.yolo_net = cv2.dnn.readNetFromONNX(yolo_model_path)

            # Set computation backend (CPU or GPU if available)
            self.yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            print(f"✅ YOLOv8n model loaded successfully from {yolo_model_path}")
            return True

        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            print("ROI enhancement will be disabled")
            return False
        
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

def get_crosswalk_roi_bounds(zones):
    """
    Get bounding rectangle for all crosswalk zones.

    Args:
        zones: Dictionary of zone_name -> CrosswalkZone objects

    Returns:
        tuple: (x, y, width, height) bounding rectangle or None if no crosswalk zones
    """
    crosswalk_zones = {name: zone for name, zone in zones.items() if 'crosswalk' in name.lower()}

    if not crosswalk_zones:
        return None

    # Collect all points from all crosswalk zones
    all_points = []
    for zone in crosswalk_zones.values():
        all_points.extend(zone.points)

    if not all_points:
        return None

    # Find bounding rectangle
    all_points = np.array(all_points)
    min_x = int(np.min(all_points[:, 0]))
    max_x = int(np.max(all_points[:, 0]))
    min_y = int(np.min(all_points[:, 1]))
    max_y = int(np.max(all_points[:, 1]))

    return (min_x, min_y, max_x - min_x, max_y - min_y)

def merge_detections(full_frame_detections, roi_detections):
    """
    Merge full frame and ROI detections, avoiding duplicates.

    Args:
        full_frame_detections: List of Detection objects from full frame
        roi_detections: List of Detection objects from ROI enhancement

    Returns:
        list: Merged list of detections with duplicates removed
    """
    if not roi_detections:
        return full_frame_detections

    # Start with full frame detections
    merged_detections = list(full_frame_detections)
    labels = get_labels(app)
    roi_object_types = get_system_setting('roi_object_types', ['person', 'bicycle'])

    # Add ROI detections that don't overlap significantly with existing ones
    for roi_det in roi_detections:
        object_type = labels[int(roi_det.category)]

        # Only merge person/bicycle detections from ROI
        if object_type not in roi_object_types:
            continue

        is_duplicate = False
        roi_box = roi_det.box

        # Check for overlap with existing detections
        for existing_det in full_frame_detections:
            existing_object_type = labels[int(existing_det.category)]

            # Only check against same object types
            if existing_object_type != object_type:
                continue

            existing_box = existing_det.box

            # Calculate IoU (Intersection over Union)
            iou = calculate_iou(roi_box, existing_box)

            # If IoU > 0.5, consider it a duplicate and keep the higher confidence one
            if iou > 0.5:
                if roi_det.conf > existing_det.conf:
                    # Replace existing with ROI detection (higher confidence)
                    merged_detections.remove(existing_det)
                    merged_detections.append(roi_det)
                is_duplicate = True
                break

        # If not a duplicate, add the ROI detection
        if not is_duplicate:
            merged_detections.append(roi_det)

    return merged_detections

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1, box2: [x, y, width, height] format

    Returns:
        float: IoU value between 0 and 1
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate intersection coordinates
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    # No intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate union area
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area

    # Avoid division by zero
    if union_area == 0:
        return 0.0

    return intersection_area / union_area

def run_enhanced_roi_detection(frame, roi_bounds, scale_factor, app_instance, args):
    """
    Run YOLOv8 detection on upscaled crosswalk ROI to catch distant pedestrians.

    Args:
        frame: Full frame (BGR format)
        roi_bounds: (x, y, width, height) of ROI in full frame
        scale_factor: Scale factor to upscale ROI
        app_instance: CrosswalkDetectionApp instance
        args: Command line arguments

    Returns:
        list: Detection objects with coordinates mapped back to full frame
    """
    roi_detections = []

    # Check if YOLO model is loaded
    if app_instance.yolo_net is None:
        return []

    try:
        x, y, w, h = roi_bounds

        # Extract and upscale ROI
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return []

        scaled_w = int(w * scale_factor)
        scaled_h = int(h * scale_factor)
        upscaled_roi = cv2.resize(roi, (scaled_w, scaled_h))

        # Prepare input for YOLO
        input_size = get_system_setting('yolo_input_size', 640)
        blob = cv2.dnn.blobFromImage(
            upscaled_roi,
            1/255.0,  # Scale factor to normalize pixel values
            (input_size, input_size),
            swapRB=True,  # OpenCV uses BGR, YOLO expects RGB
            crop=False
        )

        # Run YOLO inference
        app_instance.yolo_net.setInput(blob)
        outputs = app_instance.yolo_net.forward()

        # Parse YOLO outputs
        roi_detections = parse_yolo_outputs(
            outputs, upscaled_roi.shape, roi_bounds, scale_factor, app_instance
        )

        # Optional: Log YOLO detections (uncomment if needed for debugging)
        # if roi_detections:
        #     print(f"YOLO ROI: Found {len(roi_detections)} pedestrians in crosswalk area")

    except Exception as e:
        print(f"YOLO ROI detection error: {e}")

    return roi_detections

def parse_yolo_outputs(outputs, roi_shape, roi_bounds, scale_factor, app_instance):
    """
    Parse YOLOv8 outputs and convert to Detection objects with full-frame coordinates.

    Args:
        outputs: YOLO model output tensor
        roi_shape: Shape of the upscaled ROI (height, width, channels)
        roi_bounds: (x, y, width, height) of ROI in full frame
        scale_factor: Scale factor used for ROI upscaling
        app_instance: CrosswalkDetectionApp instance

    Returns:
        list: Detection objects mapped to full frame coordinates
    """
    detections = []

    try:
        # YOLOv8 output format can vary, let's handle both formats
        output = outputs[0]

        # Handle different output formats
        if len(output.shape) == 3:
            # Format: [1, 84, 8400] -> transpose to [1, 8400, 84]
            output = output.transpose(0, 2, 1)

        # Now should be [1, 8400, 84] format
        if len(output.shape) == 3:
            output = output[0]  # Remove batch dimension -> [8400, 84]

        # Get settings
        conf_threshold = get_system_setting('yolo_confidence_threshold', 0.3)
        nms_threshold = get_system_setting('yolo_nms_threshold', 0.4)
        person_class_id = get_system_setting('yolo_person_class_id', 0)

        # Parse detections
        boxes = []
        confidences = []
        class_ids = []

        roi_x, roi_y, roi_w, roi_h = roi_bounds
        roi_height, roi_width = roi_shape[:2]
        input_size = get_system_setting('yolo_input_size', 640)

        # YOLOv8 output format: [cx, cy, w, h, class0_conf, class1_conf, ...]
        for i in range(output.shape[0]):
            detection = output[i, :]

            # Extract bbox coordinates (normalized to input size)
            cx, cy, width, height = detection[:4]

            # Get person class confidence
            person_confidence = detection[4 + person_class_id]

            if person_confidence > conf_threshold:
                # Convert from normalized coordinates to input image pixel coordinates
                x = (cx - width/2) * input_size / input_size * roi_width
                y = (cy - height/2) * input_size / input_size * roi_height
                w = width * input_size / input_size * roi_width
                h = height * input_size / input_size * roi_height

                # Scale down from upscaled ROI to original ROI size
                x = x / scale_factor
                y = y / scale_factor
                w = w / scale_factor
                h = h / scale_factor

                # Map to full frame coordinates
                x_full = int(x + roi_x)
                y_full = int(y + roi_y)
                w_full = int(w)
                h_full = int(h)

                # Bounds checking
                if x_full >= 0 and y_full >= 0 and w_full > 0 and h_full > 0:
                    boxes.append([x_full, y_full, w_full, h_full])
                    confidences.append(float(person_confidence))
                    class_ids.append(person_class_id)

        # Apply Non-Maximum Suppression
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

            if len(indices) > 0:
                indices = indices.flatten()

                for i in indices:
                    x, y, w, h = boxes[i]
                    confidence = confidences[i]

                    # Create Detection-like object for person
                    detection = type('Detection', (), {
                        'box': [x, y, w, h],
                        'category': person_class_id,
                        'conf': confidence
                    })()

                    detections.append(detection)

    except Exception as e:
        print(f"Error parsing YOLO outputs: {e}")

    return detections

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

    # Apply per-object detection thresholds
    labels = get_labels(app_instance)
    filtered_detections = []
    for box, score, category in zip(boxes, scores, classes):
        object_type = labels[int(category)]
        object_threshold = get_detection_threshold(object_type)

        if score > object_threshold:
            filtered_detections.append(Detection(box, category, score, metadata))

    app_instance.last_detections = filtered_detections
    
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
        
        # Draw weighted detection point (75% down from top)
        center_x = x + w // 2
        center_y = y + int(h * 0.75)  # Weighted lower for better vehicle detection
        cv2.circle(frame, (center_x, center_y), radius=4, color=color, thickness=-1)

        # Draw a small marker to show this is the detection point
        cv2.circle(frame, (center_x, center_y), radius=6, color=(255, 255, 255), thickness=1)
        
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

    # Get active zone configuration info
    zone_config = get_active_zone_config()
    zone_config_name = zone_config['name']
    print(f"Using zone configuration: {zone_config_name}")
    print(f"Description: {zone_config['description']}")

    # Setup separate log files for different entity types
    app.crosswalk_monitor.setup_log_files(str(log_path), zone_config_name)

    # This must be called before instantiation of Picamera2
    app.imx500 = IMX500(imx_model)
    app.intrinsics = app.imx500.network_intrinsics
    if not app.intrinsics:
        app.intrinsics = NetworkIntrinsics()
        app.intrinsics.task = "object detection"
    elif app.intrinsics.task != "object detection":
        print("Network is not an object detection task", file=sys.stderr)
        exit()

    # Load YOLO model for ROI enhancement
    if get_system_setting('enable_roi_detection', True):
        print("Loading YOLOv8 model for ROI enhancement...")
        app.load_yolo_model()

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
    print(f"Default Detection Threshold: {args.threshold}")

    # Show per-object thresholds
    object_thresholds = get_system_setting('object_thresholds', {})
    if object_thresholds:
        print("Per-object Detection Thresholds:")
        for obj_type, threshold in object_thresholds.items():
            print(f"  {obj_type}: {threshold}")

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

            # Enhanced ROI detection for crosswalk areas
            roi_detections = []
            if get_system_setting('enable_roi_detection', True):
                try:
                    # Get current frame for ROI processing
                    frame = app.picam2.capture_array()
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    # Get ROI bounds for crosswalk zones
                    roi_bounds = get_crosswalk_roi_bounds(app.crosswalk_monitor.zones)

                    if roi_bounds is not None:
                        scale_factor = get_system_setting('roi_scale_factor', 2.5)
                        roi_detections = run_enhanced_roi_detection(
                            frame, roi_bounds, scale_factor, app, args
                        )

                        # Merge ROI detections with full frame detections
                        last_results = merge_detections(last_results, roi_detections)

                except Exception as e:
                    print(f"ROI detection error: {e}")

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

        # Print detailed statistics
        print("\nCrossing Statistics:")
        for stat_name, count in app.crosswalk_monitor.stats.items():
            print(f"  {stat_name}: {count}")

        print(f"\nLogs saved to:")
        for log_type, log_path in app.crosswalk_monitor.log_files.items():
            if log_path:
                print(f"  {log_type}: {log_path}")

        app.picam2.stop()

if __name__ == "__main__":
    main()
