#!/usr/bin/env python3
"""
find_borders.py - Interactive camera stream for finding zone coordinates

This tool streams camera output to a web interface and shows coordinates 
when you click on the image. Use this to determine exact pixel coordinates
for setting up crosswalk zones in your main detection system.
"""

import cv2
import threading
import time
import argparse
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from picamera2 import Picamera2

# Local imports
from metasettings import (
    get_active_zone_config,
    get_system_setting,
    set_external_zone_file
)

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

class CoordinateFinder:
    """Camera streaming app for finding coordinates by clicking"""

    def __init__(self):
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.clicked_points = []
        self.picam2 = None
        self.frame_width = 1280
        self.frame_height = 720
        self.zones = {}
        
    def get_latest_frame(self):
        """Thread-safe method to get the latest frame"""
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def update_frame(self, frame):
        """Thread-safe method to update the latest frame"""
        with self.frame_lock:
            self.latest_frame = frame
            
    def add_clicked_point(self, x, y):
        """Add a clicked coordinate point"""
        self.clicked_points.append((x, y))
        print(f"Clicked point: ({x}, {y})")
        # Keep only last 10 points to avoid clutter
        if len(self.clicked_points) > 10:
            self.clicked_points.pop(0)
    
    def clear_points(self):
        """Clear all clicked points"""
        self.clicked_points.clear()
        print("Cleared all points")

    def setup_zones_from_metasettings(self):
        """
        Load zone definitions from metasettings and convert to absolute coordinates.
        """
        zone_config = get_active_zone_config()
        zone_data = zone_config['zones']

        self.zones = {}
        for key, value in zone_data.items():
            name = value["name"]
            norm_points = value["points"]
            abs_points = [
                (int(x * self.frame_width), int(y * self.frame_height))
                for (x, y) in norm_points
            ]
            self.zones[key] = CrosswalkZone(name, abs_points)

        print(f"Loaded {len(self.zones)} zones from configuration: {zone_config['name']}")
        for zone_name, zone in self.zones.items():
            print(f"  - {zone_name}: {zone.name}")

class CoordinateHandler(BaseHTTPRequestHandler):
    """HTTP handler for coordinate finding interface"""
    
    def __init__(self, app_instance, *args, **kwargs):
        self.app = app_instance
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        if self.path == '/':
            self._serve_html()
        elif self.path == '/stream.mjpg':
            self._serve_video_stream()
        elif self.path.startswith('/click'):
            self._handle_click()
        elif self.path == '/clear':
            self._handle_clear()
    
    def _handle_click(self):
        """Handle coordinate click from web interface"""
        parsed_url = urlparse(self.path)
        params = parse_qs(parsed_url.query)
        
        try:
            x = int(params['x'][0])
            y = int(params['y'][0])
            self.app.add_clicked_point(x, y)
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = f'{{"status": "ok", "x": {x}, "y": {y}}}'
            self.wfile.write(response.encode())
        except (KeyError, ValueError, IndexError):
            self.send_response(400)
            self.end_headers()
    
    def _handle_clear(self):
        """Handle clear points request"""
        self.app.clear_points()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(b'{"status": "cleared"}')
    
    def _serve_html(self):
        """Serve the coordinate finding interface"""
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Zone Coordinate Finder</title>
            <style>
                body {{ 
                    font-family: Arial, sans-serif; 
                    background: #1a1a1a; 
                    color: white; 
                    margin: 0; 
                    padding: 20px; 
                }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                h1 {{ color: #ffcc00; text-align: center; }}
                .video-container {{ 
                    text-align: center; 
                    margin: 20px 0; 
                    position: relative;
                    display: inline-block;
                }}
                .clickable-image {{
                    border: 2px solid #555;
                    border-radius: 8px;
                    cursor: crosshair;
                    width: 960px;
                    height: 720px;
                    max-width: 100%;
                    object-fit: fill;
                }}
                .instructions {{ 
                    background: #333; 
                    padding: 15px; 
                    border-radius: 8px; 
                    margin: 20px 0; 
                }}
                .controls {{ text-align: center; margin: 10px 0; }}
                button {{ 
                    background: #ffcc00; 
                    color: black; 
                    border: none; 
                    padding: 10px 20px; 
                    border-radius: 5px; 
                    cursor: pointer; 
                    margin: 5px;
                }}
                button:hover {{ background: #ffd700; }}
                .coordinates {{ 
                    background: #222; 
                    padding: 10px; 
                    border-radius: 5px; 
                    font-family: monospace; 
                    margin: 10px 0;
                    max-height: 200px;
                    overflow-y: auto;
                }}
                .grid-overlay {{
                    position: absolute;
                    top: 0;
                    left: 0;
                    pointer-events: none;
                    opacity: 0.3;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Zone Coordinate Finder</h1>
                <div class="instructions">
                    <h3>Instructions:</h3>
                    <ul>
                        <li><strong>Click on the image</strong> to get pixel coordinates</li>
                        <li>Zone boundaries are overlaid in different colors (if configured)</li>
                        <li>Use these coordinates to define zones in your crosswalk detector</li>
                        <li>Camera resolution: {self.app.frame_width}x{self.app.frame_height}</li>
                        <li>For normalized coordinates (0.0-1.0), divide X by {self.app.frame_width} and Y by {self.app.frame_height}</li>
                    </ul>
                </div>
                <div class="video-container">
                    <img id="stream" class="clickable-image" src="/stream.mjpg" alt="Camera Feed">
                </div>
                <div class="controls">
                    <button onclick="clearPoints()">Clear Points</button>
                    <button onclick="toggleGrid()">Show Grid</button>
                    <button onclick="copyCoordinates()">Copy Last Point</button>
                </div>
                <div class="coordinates" id="coordinates">
                    <strong>Clicked Coordinates:</strong><br>
                    Click on the image above to see coordinates here...
                </div>
                <div class="stats">
                    <strong>Loaded Zones:</strong><br>
                    {self._format_zones_info()}
                </div>
            </div>
            <script>
                let showGrid = false;
                let lastClickedPoint = null;
                let gridCanvas = null;

                // Wait for page to load
                document.addEventListener('DOMContentLoaded', function() {{
                    setupEventListeners();
                    createGridCanvas();
                }});

                function setupEventListeners() {{
                    const streamImg = document.getElementById('stream');
                    if (!streamImg) return;

                    streamImg.addEventListener('click', function(e) {{
                        const rect = this.getBoundingClientRect();
                        const scaleX = {self.app.frame_width} / rect.width;
                        const scaleY = {self.app.frame_height} / rect.height;

                        const x = Math.round((e.clientX - rect.left) * scaleX);
                        const y = Math.round((e.clientY - rect.top) * scaleY);

                        console.log('Click detected:', x, y);

                        // Send click to server
                        fetch(`/click?x=${{x}}&y=${{y}}`)
                            .then(response => {{
                                if (!response.ok) {{
                                    throw new Error('Network response was not ok');
                                }}
                                return response.json();
                            }})
                            .then(data => {{
                                console.log('Server response:', data);
                                if (data.status === 'ok') {{
                                    lastClickedPoint = {{x: data.x, y: data.y}};
                                    updateCoordinatesDisplay();
                                }}
                            }})
                            .catch(err => {{
                                console.error('Error sending click:', err);
                                // Still update display even if server fails
                                lastClickedPoint = {{x: x, y: y}};
                                updateCoordinatesDisplay();
                            }});
                    }});

                    // Handle image load to resize grid
                    streamImg.addEventListener('load', function() {{
                        updateGridSize();
                    }});
                }}

                function createGridCanvas() {{
                    const videoContainer = document.querySelector('.video-container');
                    if (!videoContainer) return;

                    gridCanvas = document.createElement('canvas');
                    gridCanvas.className = 'grid-overlay';
                    gridCanvas.style.position = 'absolute';
                    gridCanvas.style.top = '0';
                    gridCanvas.style.left = '0';
                    gridCanvas.style.pointerEvents = 'none';
                    gridCanvas.style.opacity = '0.5';
                    gridCanvas.style.display = 'none';

                    videoContainer.appendChild(gridCanvas);
                    updateGridSize();
                }}

                function updateGridSize() {{
                    if (!gridCanvas) return;

                    const streamImg = document.getElementById('stream');
                    if (!streamImg) return;

                    const rect = streamImg.getBoundingClientRect();
                    gridCanvas.width = rect.width;
                    gridCanvas.height = rect.height;
                    gridCanvas.style.width = rect.width + 'px';
                    gridCanvas.style.height = rect.height + 'px';

                    if (showGrid) {{
                        drawGrid();
                    }}
                }}

                function drawGrid() {{
                    if (!gridCanvas) return;

                    const ctx = gridCanvas.getContext('2d');
                    ctx.clearRect(0, 0, gridCanvas.width, gridCanvas.height);

                    // Calculate grid spacing based on 0.1 increments of actual frame dimensions
                    const xSpacing = gridCanvas.width * 0.1;  // 10% of display width
                    const ySpacing = gridCanvas.height * 0.1; // 10% of display height

                    // Style for intersection points
                    ctx.fillStyle = '#00ff00';
                    ctx.strokeStyle = '#00ff00';
                    ctx.lineWidth = 2;

                    // Style for text - bigger and more legible
                    ctx.font = 'bold 16px Arial';
                    ctx.textAlign = 'left';
                    ctx.textBaseline = 'top';

                    // Add text shadow/outline for better legibility
                    ctx.shadowColor = '#000000';
                    ctx.shadowBlur = 3;
                    ctx.shadowOffsetX = 1;
                    ctx.shadowOffsetY = 1;

                    // Draw intersection points and labels
                    for (let i = 0; i <= 10; i++) {{
                        for (let j = 0; j <= 10; j++) {{
                            const x = i * xSpacing;
                            const y = j * ySpacing;

                            // Skip the origin point (0,0) to avoid clutter
                            if (i === 0 && j === 0) continue;

                            // Draw intersection point (small circle)
                            ctx.beginPath();
                            ctx.arc(x, y, 4, 0, 2 * Math.PI);
                            ctx.fill();

                            // Calculate integer coordinates (0-10 scale)
                            const intX = i;
                            const intY = j;

                            // Draw coordinate label with integer coordinates
                            const label = `(${{intX}},${{intY}})`;

                            // Position text to avoid overlapping with the point
                            let textX = x + 8;
                            let textY = y - 10;

                            // Adjust position if near edges to keep text visible
                            if (textX + 60 > gridCanvas.width) textX = x - 60;
                            if (textY < 0) textY = y + 8;

                            // Draw coordinate text
                            ctx.fillText(label, textX, textY);
                        }}
                    }}

                    // Reset shadow for future drawing
                    ctx.shadowColor = 'transparent';
                    ctx.shadowBlur = 0;
                    ctx.shadowOffsetX = 0;
                    ctx.shadowOffsetY = 0;
                }}

                function updateCoordinatesDisplay() {{
                    if (lastClickedPoint) {{
                        const normalizedX = (lastClickedPoint.x / {self.app.frame_width}).toFixed(3);
                        const normalizedY = (lastClickedPoint.y / {self.app.frame_height}).toFixed(3);

                        const coordsDiv = document.getElementById('coordinates');
                        if (coordsDiv) {{
                            const timestamp = new Date().toLocaleTimeString();
                            const newCoord = `<br>[${{timestamp}}] Pixel: (${{lastClickedPoint.x}}, ${{lastClickedPoint.y}}) | Normalized: (${{normalizedX}}, ${{normalizedY}})`;

                            // If this is the first coordinate, replace the placeholder text
                            if (coordsDiv.innerHTML.includes('Click on the image above')) {{
                                coordsDiv.innerHTML = '<strong>Clicked Coordinates:</strong>' + newCoord;
                            }} else {{
                                coordsDiv.innerHTML += newCoord;
                            }}
                            coordsDiv.scrollTop = coordsDiv.scrollHeight;
                        }}
                    }}
                }}

                function clearPoints() {{
                    fetch('/clear')
                        .then(response => response.json())
                        .then(data => {{
                            const coordsDiv = document.getElementById('coordinates');
                            if (coordsDiv) {{
                                coordsDiv.innerHTML = '<strong>Clicked Coordinates:</strong><br>Cleared all points...';
                            }}
                            lastClickedPoint = null;
                        }})
                        .catch(err => console.error('Error clearing points:', err));
                }}

                function toggleGrid() {{
                    showGrid = !showGrid;
                    if (gridCanvas) {{
                        gridCanvas.style.display = showGrid ? 'block' : 'none';
                        if (showGrid) {{
                            updateGridSize();
                            drawGrid();
                        }}
                    }}

                    const button = event.target;
                    button.textContent = showGrid ? 'Hide Grid' : 'Show Grid';
                    button.style.background = showGrid ? '#ff6b6b' : '#ffcc00';
                }}

                function copyCoordinates() {{
                    if (lastClickedPoint) {{
                        const text = `(${{lastClickedPoint.x}}, ${{lastClickedPoint.y}})`;
                        navigator.clipboard.writeText(text).then(() => {{
                            alert('Coordinates copied to clipboard: ' + text);
                        }}).catch(err => {{
                            // Fallback for older browsers
                            console.error('Clipboard copy failed:', err);
                            prompt('Copy these coordinates:', text);
                        }});
                    }} else {{
                        alert('No coordinates to copy. Click on the image first.');
                    }}
                }}

                // Auto-refresh if stream fails
                document.getElementById('stream').addEventListener('error', function() {{
                    console.log('Stream error, reloading...');
                    setTimeout(() => location.reload(), 2000);
                }});

                // Handle window resize
                window.addEventListener('resize', function() {{
                    setTimeout(updateGridSize, 100);
                }});
            </script>
        </body>
        </html>
        """
        self.wfile.write(html.encode())

    def _format_zones_info(self):
        """Format zone information for HTML display"""
        if not self.app.zones:
            return "No zones configured"

        zone_info = []
        for zone_name, zone in self.app.zones.items():
            zone_info.append(f"{zone_name}: {zone.name}")
        return "<br>".join(zone_info)
    
    def _serve_video_stream(self):
        """Serve the MJPEG video stream with coordinate overlay"""
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
        self.end_headers()

        while True:
            try:
                frame = self.app.get_latest_frame()
                if frame is not None:
                    # Draw zones first (so they appear behind points)
                    self._draw_zones_on_frame(frame)

                    # Draw clicked points on the frame
                    for i, (x, y) in enumerate(self.app.clicked_points):
                        # Draw point
                        cv2.circle(frame, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
                        # Draw label with coordinates
                        label = f"({x},{y})"
                        cv2.putText(frame, label, (x + 10, y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        # Draw point number
                        cv2.putText(frame, str(i + 1), (x - 15, y + 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    
                    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    frame_data = jpeg.tobytes()
                else:
                    # Create placeholder frame
                    placeholder = cv2.zeros((self.app.frame_height, self.app.frame_width, 3), dtype=cv2.uint8)
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
                
                time.sleep(1.0 / 10)  # 10 FPS
                
            except Exception as e:
                print(f"Streaming error: {e}")
                break

    def _draw_zones_on_frame(self, frame):
        """
        Draw zone boundaries on the video frame for visualization.

        Args:
            frame: OpenCV frame array to draw on
        """
        # Get zone colors from system settings, or use defaults
        zone_colors = get_system_setting('zone_colors', {})
        default_colors = {
            'crosswalk': (0, 255, 255),  # Yellow
            'north_lane': (255, 0, 0),   # Blue
            'south_lane': (0, 0, 255),   # Red
            'sidewalk': (128, 128, 128), # Gray
            'default': (255, 255, 255)   # White
        }

        for zone_name, zone in self.app.zones.items():
            # Get color for this zone type
            color = zone_colors.get(zone_name, default_colors.get(zone_name, default_colors['default']))

            # Draw zone boundary
            cv2.polylines(frame, [zone.points], True, color, 2)

            # Draw corner points
            for point in zone.points:
                cv2.circle(frame, tuple(point), radius=4, color=color, thickness=-1)

            # Label the zone
            if len(zone.points) > 0:
                cv2.putText(frame, zone_name.upper(),
                           tuple(zone.points[0]), cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, color, 2)

class WebServer:
    """Web server for coordinate finding interface"""
    
    def __init__(self, app_instance):
        self.app = app_instance
        
    def start(self, port=8081):
        """Start the web server in a separate thread"""
        def run_server():
            try:
                server = HTTPServer(('0.0.0.0', port), lambda *args: CoordinateHandler(self.app, *args))
                print(f"Coordinate finder available at http://localhost:{port}")
                print(f"Network access: http://[your_pi_ip]:{port}")
                server.serve_forever()
            except Exception as e:
                print(f"Web server error: {e}")
        
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        return thread

def get_args():
    parser = argparse.ArgumentParser(description="Find zone coordinates by clicking on camera stream")
    parser.add_argument("--port", type=int, default=8081, help="Web server port (default: 8081)")
    parser.add_argument("--width", type=int, default=1280, help="Camera width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Camera height (default: 720)")
    parser.add_argument("--zone-config", type=str, help="Zone configuration file to load (e.g., 'chapel-crosswalk')")
    return parser.parse_args()

def main():
    """Main function"""
    args = get_args()

    # Set external zone file if specified
    if args.zone_config:
        set_external_zone_file(args.zone_config)
        print(f"Using external zone configuration: {args.zone_config}")

    print("Zone Coordinate Finder")
    print("=" * 30)

    # Initialize the coordinate finder app
    app = CoordinateFinder()
    app.frame_width = args.width
    app.frame_height = args.height

    # Setup zones from metasettings
    try:
        app.setup_zones_from_metasettings()
    except Exception as e:
        print(f"Warning: Could not load zone configuration: {e}")
        print("Continuing without zone overlay...")
    
    # Initialize camera
    print("Initializing camera...")
    app.picam2 = Picamera2()
    config = app.picam2.create_preview_configuration(
        main={"size": (args.width, args.height)}
    )
    
    print(f"Camera resolution: {args.width}x{args.height}")
    
    # Start web server
    web_server = WebServer(app)
    web_server.start(args.port)
    
    print("Starting camera...")
    app.picam2.start(config, show_preview=False)
    print("Camera started successfully!")
    print()
    print("Instructions:")
    print("1. Open your web browser and go to the URL shown above")
    print("2. Click anywhere on the camera image to get coordinates")
    print("3. Use these coordinates to set up zones in your crosswalk detector")
    print("4. Press Ctrl+C to stop")
    print()
    
    try:
        while True:
            # Capture frame and update the web stream
            frame = app.picam2.capture_array()
            # Convert RGB to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            app.update_frame(frame)
            time.sleep(0.1)  # Small delay to prevent excessive CPU usage
            
    except KeyboardInterrupt:
        print("\nCoordinate finder stopped")
        print("Clicked points summary:")
        for i, (x, y) in enumerate(app.clicked_points):
            norm_x = x / args.width
            norm_y = y / args.height
            print(f"  Point {i+1}: ({x}, {y}) | Normalized: ({norm_x:.3f}, {norm_y:.3f})")
        app.picam2.stop()

if __name__ == "__main__":
    main()