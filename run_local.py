"""
LiteDepth - Local Windows Runner
Real-Time 3D Perception for Industrial Safety

This is a standalone script to run the LiteDepth system locally on Windows.
It processes video files and provides WebSocket streaming for the frontend.

Usage:
    1. Install dependencies: pip install -r requirements.txt
    2. Run the script: python run_local.py
    3. Open websocket.html in your browser
    4. Connect to ws://localhost:8765
"""

import asyncio
import os
import sys
import time
import logging
import traceback
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import websockets
from ultralytics import YOLO

# ==================== CONFIGURATION ====================

# Video input - Change this to your video file path
VIDEO_PATH = None  # Will prompt you to select if None

# Model configuration
YOLO_MODEL_NAME = "yolov8s.pt"  # Will download automatically if not present
DEPTH_MODEL_TYPE = "vits"  # Options: vits, vitb, vitl

# Server configuration
WEBSOCKET_PORT = 8765

# Processing configuration
CONFIDENCE_THRESHOLD = 0.4
TARGET_CLASSES = []  # Empty = all classes

# Output configuration
ENABLE_VIDEO_WRITING = True
OUTPUT_VIDEO_FILENAME = "processed_output.mp4"

# ==================== LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ==================== GLOBALS ====================
_depth_model = None
_yolo_model = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
active_connections = set()

# Color constants
PDF_COLOR_CLOSE = (0, 0, 255)    # Red
PDF_COLOR_MEDIUM = (0, 255, 255) # Yellow
PDF_COLOR_FAR = (0, 255, 0)      # Green
COLOR_TEXT_BOX = (255, 255, 255)

# Thresholds
PDF_RAW_VAL_CLOSE_THRESHOLD = 0.6
PDF_RAW_VAL_MEDIUM_THRESHOLD = 0.3

# BEV Configuration
BEV_HEIGHT = 240
BEV_WIDTH = 320
BEV_BACKGROUND_COLOR = (30, 30, 30)
EGO_VEHICLE_COLOR_BEV = (0, 220, 0)
EGO_VEHICLE_RECT_W = 20
EGO_VEHICLE_RECT_H = 12
VEHICLE_POSITION_BEV = (BEV_WIDTH // 2, BEV_HEIGHT - 20)
BEV_DOT_RADIUS_DANGER = 8
BEV_DOT_RADIUS_CAUTION = 6
BEV_DOT_RADIUS_FAR = 4
BEV_DOT_RADIUS_MIN = 2
BEV_Y_SCALE_RAW_FRACTION = BEV_HEIGHT * 0.90
BEV_X_SCALE_FACTOR = BEV_WIDTH * 0.8
BEV_DOT_BORDER_COLOR = (200, 200, 200)
BEV_DOT_BORDER_THICKNESS = 1
BLINK_FRAME_INTERVAL = 10
MAX_TRAIL_LENGTH = 5
TRACKING_MAX_DIST = 35

# Tracking globals
BEV_OBJECT_TRACKS = {}
NEXT_TRACK_ID = 0


# ==================== DEPTH MODEL LOADING ====================

def download_depth_model():
    """Download and setup the DepthAnythingV2 model."""
    global _depth_model
    
    logging.info("Setting up Depth Estimation Model...")
    
    # Clone the repository if not exists
    repo_dir = Path(__file__).parent / "Python-Depth-Est-AV2"
    
    if not repo_dir.exists():
        logging.info("Cloning DepthAnythingV2 repository...")
        os.system(f'git clone https://github.com/computervisionpro/Python-Depth-Est-AV2.git "{repo_dir}"')
    
    # Add to path
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    
    # Create checkpoints directory
    checkpoint_dir = repo_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Model configurations
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    
    # Checkpoint paths
    checkpoint_name = f"depth_anything_v2_{DEPTH_MODEL_TYPE}.pth"
    checkpoint_path = checkpoint_dir / checkpoint_name
    
    # Download if not exists
    if not checkpoint_path.exists():
        hf_model_name = {'vits': 'Small', 'vitb': 'Base', 'vitl': 'Large'}.get(DEPTH_MODEL_TYPE, 'Small')
        download_url = f"https://huggingface.co/depth-anything/Depth-Anything-V2-{hf_model_name}/resolve/main/{checkpoint_name}"
        logging.info(f"Downloading depth model checkpoint from {download_url}...")
        
        import urllib.request
        urllib.request.urlretrieve(download_url, checkpoint_path)
        logging.info("Download complete.")
    
    # Load model
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
        
        config = model_configs[DEPTH_MODEL_TYPE]
        _depth_model = DepthAnythingV2(
            encoder=config['encoder'],
            features=config['features'],
            out_channels=config['out_channels']
        )
        
        state_dict = torch.load(str(checkpoint_path), map_location='cpu')
        _depth_model.load_state_dict(state_dict, strict=False)
        _depth_model = _depth_model.to(DEVICE).eval()
        
        logging.info(f"✅ Depth Model '{DEPTH_MODEL_TYPE}' loaded successfully on {DEVICE}")
        return True
        
    except Exception as e:
        logging.error(f"❌ Failed to load depth model: {e}")
        traceback.print_exc()
        return False


def load_yolo_model():
    """Load the YOLO object detection model."""
    global _yolo_model
    
    logging.info(f"Loading YOLO model: {YOLO_MODEL_NAME}...")
    
    try:
        _yolo_model = YOLO(YOLO_MODEL_NAME)
        logging.info(f"✅ YOLO Model '{YOLO_MODEL_NAME}' loaded successfully")
        return True
    except Exception as e:
        logging.error(f"❌ Failed to load YOLO model: {e}")
        return False


# ==================== HELPER FUNCTIONS ====================

def colorize_relative_depth(raw_depth_map, colormap=cv2.COLORMAP_INFERNO):
    """Apply colormap to depth map for visualization."""
    if raw_depth_map is None:
        return None
    
    vis_map = raw_depth_map.astype(np.float32)
    min_val, max_val = np.min(vis_map), np.max(vis_map)
    
    if max_val > min_val:
        normalized = (vis_map - min_val) / (max_val - min_val)
        depth_uint8 = (normalized * 255).astype(np.uint8)
    else:
        depth_uint8 = np.zeros_like(vis_map, dtype=np.uint8)
    
    return cv2.applyColorMap(depth_uint8, colormap)


def get_raw_depth_at_center(raw_depth_map, box_coords):
    """Get depth value at the center of a bounding box."""
    if raw_depth_map is None:
        return None
    
    x1, y1, x2, y2 = map(int, box_coords)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    
    try:
        h, w = raw_depth_map.shape[:2]
        sy = max(0, min(h - 1, int(round(cy))))
        sx = max(0, min(w - 1, int(round(cx))))
        val = raw_depth_map[sy, sx]
        return val if np.isfinite(val) else None
    except Exception:
        return None


def draw_ar_bounding_boxes(frame, yolo_results, raw_depth_map, min_raw, max_raw, names):
    """Draw AR-style bounding boxes with depth coloring."""
    if yolo_results is None or len(yolo_results) == 0 or yolo_results[0].boxes is None:
        return frame
    
    boxes_data = yolo_results[0].boxes
    
    for i in range(len(boxes_data.xyxy)):
        box = boxes_data.xyxy[i].cpu().numpy()
        conf = boxes_data.conf[i].cpu().numpy()
        cls_id = int(boxes_data.cls[i].cpu().numpy())
        class_name = names[cls_id]
        
        if TARGET_CLASSES and class_name not in TARGET_CLASSES:
            continue
        if conf < CONFIDENCE_THRESHOLD:
            continue
        
        x1, y1, x2, y2 = map(int, box)
        obj_raw_depth = get_raw_depth_at_center(raw_depth_map, box)
        
        box_color = PDF_COLOR_FAR
        depth_text = "(Depth N/A)"
        
        if obj_raw_depth is not None and max_raw > min_raw:
            depth_text = f"({obj_raw_depth:.2e})"
            d_range = max_raw - min_raw
            close_thr = min_raw + d_range * PDF_RAW_VAL_CLOSE_THRESHOLD
            med_thr = min_raw + d_range * PDF_RAW_VAL_MEDIUM_THRESHOLD
            
            if obj_raw_depth >= close_thr:
                box_color = PDF_COLOR_CLOSE
            elif obj_raw_depth >= med_thr:
                box_color = PDF_COLOR_MEDIUM
        
        txt = f"{class_name}({conf:.2f}){depth_text}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tbg_y1 = max(0, y1 - th - 4)
        ty_pos = max(th, y1 - 4)
        
        cv2.rectangle(frame, (x1, tbg_y1), (x1 + tw, y1), box_color, -1)
        cv2.putText(frame, txt, (x1, ty_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT_BOX, 1, cv2.LINE_AA)
    
    return frame


def draw_depth_lines(frame, yolo_results, raw_depth_map, min_raw, max_raw):
    """Draw colored lines from camera to detected objects."""
    if yolo_results is None or len(yolo_results) == 0 or yolo_results[0].boxes is None:
        return
    
    h, w = frame.shape[:2]
    cam_ox, cam_oy = w // 2, h - 1
    
    boxes_data = yolo_results[0].boxes
    
    for i in range(len(boxes_data.xyxy)):
        if boxes_data.conf[i] < CONFIDENCE_THRESHOLD:
            continue
        
        x1, y1, x2, y2 = map(int, boxes_data.xyxy[i].cpu().numpy())
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        depth_val = get_raw_depth_at_center(raw_depth_map, (x1, y1, x2, y2))
        
        if depth_val is not None and max_raw > min_raw:
            d_range = max_raw - min_raw
            close_thr = min_raw + d_range * PDF_RAW_VAL_CLOSE_THRESHOLD
            med_thr = min_raw + d_range * PDF_RAW_VAL_MEDIUM_THRESHOLD
            
            if depth_val >= close_thr:
                line_color = PDF_COLOR_CLOSE
            elif depth_val >= med_thr:
                line_color = PDF_COLOR_MEDIUM
            else:
                line_color = PDF_COLOR_FAR
            
            cv2.line(frame, (cam_ox, cam_oy), (int(cx), int(cy)), line_color, 2)


def update_bev_tracks(current_detections):
    """Update object tracking for BEV display."""
    global BEV_OBJECT_TRACKS, NEXT_TRACK_ID
    
    if not BEV_OBJECT_TRACKS:
        for det in current_detections:
            BEV_OBJECT_TRACKS[NEXT_TRACK_ID] = deque(
                [(det['x_base'], det['y_base'], det['color'])],
                maxlen=MAX_TRAIL_LENGTH
            )
            det['track_id'] = NEXT_TRACK_ID
            NEXT_TRACK_ID += 1
        return current_detections
    
    unmatched_track_ids = list(BEV_OBJECT_TRACKS.keys())
    unmatched_detections = list(range(len(current_detections)))
    
    for track_id in list(unmatched_track_ids):
        last_x, last_y, _ = BEV_OBJECT_TRACKS[track_id][-1]
        best_dist = float('inf')
        best_det_idx = -1
        
        for det_idx in list(unmatched_detections):
            det = current_detections[det_idx]
            dist = np.sqrt((last_x - det['x_base'])**2 + (last_y - det['y_base'])**2)
            if dist < TRACKING_MAX_DIST and dist < best_dist:
                best_dist = dist
                best_det_idx = det_idx
        
        if best_det_idx != -1:
            matched_det = current_detections[best_det_idx]
            BEV_OBJECT_TRACKS[track_id].append(
                (matched_det['x_base'], matched_det['y_base'], matched_det['color'])
            )
            matched_det['track_id'] = track_id
            unmatched_track_ids.remove(track_id)
            unmatched_detections.remove(best_det_idx)
    
    for track_id in unmatched_track_ids:
        del BEV_OBJECT_TRACKS[track_id]
    
    for det_idx in unmatched_detections:
        det = current_detections[det_idx]
        BEV_OBJECT_TRACKS[NEXT_TRACK_ID] = deque(
            [(det['x_base'], det['y_base'], det['color'])],
            maxlen=MAX_TRAIL_LENGTH
        )
        det['track_id'] = NEXT_TRACK_ID
        NEXT_TRACK_ID += 1
    
    return current_detections


def create_enhanced_bev(yolo_results, raw_depth_map, min_raw, max_raw, names, frame_width, frame_number):
    """Create enhanced Bird's Eye View display."""
    global BEV_OBJECT_TRACKS
    
    bev_image = np.full((BEV_HEIGHT, BEV_WIDTH, 3), BEV_BACKGROUND_COLOR, dtype=np.uint8)
    ego_center_x, ego_center_y = VEHICLE_POSITION_BEV
    
    # Draw grid
    for i in range(1, 6):
        line_y = ego_center_y - int((BEV_HEIGHT * 0.9 / 5) * i)
        if line_y < 0:
            break
        cv2.line(bev_image, (0, line_y), (BEV_WIDTH, line_y), (60, 60, 60), 1)
    
    # Draw ego vehicle
    cv2.rectangle(bev_image,
                  (ego_center_x - EGO_VEHICLE_RECT_W // 2, ego_center_y - EGO_VEHICLE_RECT_H // 2),
                  (ego_center_x + EGO_VEHICLE_RECT_W // 2, ego_center_y + EGO_VEHICLE_RECT_H // 2),
                  EGO_VEHICLE_COLOR_BEV, -1)
    cv2.line(bev_image,
             (ego_center_x, ego_center_y - EGO_VEHICLE_RECT_H // 2),
             (ego_center_x, ego_center_y - EGO_VEHICLE_RECT_H // 2 - 6),
             EGO_VEHICLE_COLOR_BEV, 3)
    
    current_detections = []
    
    if not (yolo_results and len(yolo_results) > 0 and yolo_results[0].boxes and raw_depth_map is not None):
        # Draw existing trails
        for track_id, trail in list(BEV_OBJECT_TRACKS.items()):
            for i, (tx, ty, tcolor) in enumerate(trail):
                alpha = 1.0 - ((len(trail) - 1 - i) / MAX_TRAIL_LENGTH)
                trail_color = (int(tcolor[0] * alpha), int(tcolor[1] * alpha), int(tcolor[2] * alpha))
                cv2.circle(bev_image, (tx, ty), max(1, BEV_DOT_RADIUS_MIN - 1), trail_color, -1)
        return bev_image
    
    boxes_data = yolo_results[0].boxes
    
    for i in range(len(boxes_data.xyxy)):
        box = boxes_data.xyxy[i].cpu().numpy()
        conf = boxes_data.conf[i].cpu().numpy()
        cls_id = int(boxes_data.cls[i].cpu().numpy())
        class_name = names[cls_id]
        
        if TARGET_CLASSES and class_name not in TARGET_CLASSES:
            continue
        if conf < CONFIDENCE_THRESHOLD:
            continue
        
        x1, _, x2, _ = map(int, box)
        center_x = (x1 + x2) // 2
        obj_depth = get_raw_depth_at_center(raw_depth_map, box)
        
        dot_color = PDF_COLOR_FAR
        dot_radius = BEV_DOT_RADIUS_FAR
        bev_y_fraction = 1.0
        
        if obj_depth is not None and max_raw > min_raw:
            depth_range = max_raw - min_raw
            close_thresh = min_raw + depth_range * PDF_RAW_VAL_CLOSE_THRESHOLD
            med_thresh = min_raw + depth_range * PDF_RAW_VAL_MEDIUM_THRESHOLD
            
            if obj_depth >= close_thresh:
                dot_color = PDF_COLOR_CLOSE
                dot_radius = BEV_DOT_RADIUS_DANGER
            elif obj_depth >= med_thresh:
                dot_color = PDF_COLOR_MEDIUM
                dot_radius = BEV_DOT_RADIUS_CAUTION
            
            clamped_depth = np.clip(obj_depth, min_raw, max_raw)
            if depth_range > 1e-5:
                bev_y_fraction = (max_raw - clamped_depth) / depth_range
            else:
                bev_y_fraction = 0.0
        
        y_base = ego_center_y - int(bev_y_fraction * BEV_Y_SCALE_RAW_FRACTION)
        perspective_factor = 1.0 - (0.4 * bev_y_fraction)
        x_offset = ((center_x / frame_width) - 0.5) * BEV_X_SCALE_FACTOR * perspective_factor
        x_base = ego_center_x + int(x_offset)
        
        x_base = np.clip(x_base, 0, BEV_WIDTH - 1)
        y_base = np.clip(y_base, 0, BEV_HEIGHT - 1)
        
        current_detections.append({
            'x_base': x_base, 'y_base': y_base,
            'color': dot_color, 'radius': dot_radius,
            'raw_depth': obj_depth, 'class_name': class_name
        })
    
    update_bev_tracks(current_detections)
    
    # Draw tracks and dots
    for track_id, trail in list(BEV_OBJECT_TRACKS.items()):
        # Draw trail
        for i, (tx, ty, tcolor) in enumerate(trail):
            if i < len(trail) - 1:
                alpha = 0.1 + 0.9 * (i / MAX_TRAIL_LENGTH)
                trail_color = (int(tcolor[0] * alpha), int(tcolor[1] * alpha), int(tcolor[2] * alpha))
                trail_radius = BEV_DOT_RADIUS_MIN - 2 + int((i / MAX_TRAIL_LENGTH) * 2)
                cv2.circle(bev_image, (tx, ty), max(1, trail_radius), trail_color, -1)
        
        # Draw current position
        current_x, current_y, current_color = trail[-1]
        current_radius = BEV_DOT_RADIUS_FAR
        if current_color == PDF_COLOR_CLOSE:
            current_radius = BEV_DOT_RADIUS_DANGER
        elif current_color == PDF_COLOR_MEDIUM:
            current_radius = BEV_DOT_RADIUS_CAUTION
        
        # Blinking for danger zone
        dot_visible = True
        if current_color == PDF_COLOR_CLOSE:
            if (frame_number // (BLINK_FRAME_INTERVAL // 2)) % 2 == 0:
                dot_visible = False
        
        if dot_visible:
            cv2.circle(bev_image, (current_x, current_y), current_radius, current_color, -1)
            cv2.circle(bev_image, (current_x, current_y), current_radius, BEV_DOT_BORDER_COLOR, BEV_DOT_BORDER_THICKNESS)
    
    # Draw legend
    cv2.putText(bev_image, "BEV Zones:", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT_BOX, 1)
    cv2.circle(bev_image, (10, 30), 5, PDF_COLOR_CLOSE, -1)
    cv2.putText(bev_image, "Close", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT_BOX, 1)
    cv2.circle(bev_image, (10, 50), 5, PDF_COLOR_MEDIUM, -1)
    cv2.putText(bev_image, "Medium", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT_BOX, 1)
    cv2.circle(bev_image, (10, 70), 5, PDF_COLOR_FAR, -1)
    cv2.putText(bev_image, "Far", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT_BOX, 1)
    
    return bev_image


# ==================== WEBSOCKET SERVER ====================

async def video_stream_handler(websocket):
    """Handle WebSocket connection and stream processed video."""
    global _depth_model, _yolo_model, BEV_OBJECT_TRACKS, NEXT_TRACK_ID
    
    connection_id = str(websocket.remote_address) if websocket.remote_address else f"Client-{time.time():.0f}"
    active_connections.add(websocket)
    logging.info(f"Client connected: {connection_id}. Total: {len(active_connections)}")
    
    # Reset tracking for new connection
    BEV_OBJECT_TRACKS = {}
    NEXT_TRACK_ID = 0
    
    if not all([_depth_model, _yolo_model, VIDEO_PATH]):
        logging.error("Missing model or video path")
        await websocket.close(code=1011, reason="Server not ready")
        active_connections.discard(websocket)
        return
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        logging.error(f"Cannot open video: {VIDEO_PATH}")
        await websocket.close(code=1011, reason="Cannot open video")
        active_connections.discard(websocket)
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    w_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logging.info(f"Video: {w_vid}x{h_vid} @ {fps:.1f} FPS")
    
    output_writer = None
    if ENABLE_VIDEO_WRITING:
        preview_h = max(120, h_vid // 3)
        combined_h = h_vid + preview_h
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.join(os.path.dirname(VIDEO_PATH) or ".", OUTPUT_VIDEO_FILENAME)
        output_writer = cv2.VideoWriter(output_path, fourcc, fps if fps > 0 else 10, (w_vid, combined_h))
        if output_writer.isOpened():
            logging.info(f"Saving output to: {output_path}")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            frame_count += 1
            
            if not ret or frame is None:
                logging.info(f"End of video at frame {frame_count}")
                break
            
            output_frame = frame.copy()
            
            # Depth estimation
            depth_map_raw = None
            try:
                with torch.no_grad():
                    depth_map_raw = _depth_model.infer_image(frame, input_size=518)
            except Exception as e:
                logging.error(f"Depth inference error: {e}")
            
            # YOLO detection
            yolo_results = None
            try:
                yolo_results = _yolo_model(frame, device=DEVICE, verbose=False, conf=CONFIDENCE_THRESHOLD)
            except Exception as e:
                logging.error(f"YOLO inference error: {e}")
            
            # Get depth range
            min_raw, max_raw = 0, 1
            if depth_map_raw is not None:
                finite_map = depth_map_raw[np.isfinite(depth_map_raw)]
                if finite_map.size > 0:
                    min_raw = np.min(finite_map)
                    max_raw = np.max(finite_map)
            
            # Draw depth lines
            draw_depth_lines(output_frame, yolo_results, depth_map_raw, min_raw, max_raw)
            
            # Draw AR bounding boxes
            output_frame = draw_ar_bounding_boxes(
                output_frame, yolo_results, depth_map_raw, min_raw, max_raw, _yolo_model.names
            )
            
            # Create BEV
            bev_display = create_enhanced_bev(
                yolo_results, depth_map_raw, min_raw, max_raw, _yolo_model.names, w_vid, frame_count
            )
            
            # Colorize depth
            depth_colored = colorize_relative_depth(depth_map_raw)
            if depth_colored is None:
                depth_colored = np.zeros((h_vid, w_vid, 3), dtype=np.uint8)
                cv2.putText(depth_colored, "Depth N/A", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif depth_colored.shape[:2] != (h_vid, w_vid):
                depth_colored = cv2.resize(depth_colored, (w_vid, h_vid))
            
            # Combine displays
            preview_h = max(120, h_vid // 3)
            preview_w = w_vid // 2
            
            bev_preview = cv2.resize(bev_display, (preview_w, preview_h))
            depth_preview = cv2.resize(depth_colored, (preview_w, preview_h))
            aux_row = np.hstack((bev_preview, depth_preview))
            
            if aux_row.shape[1] != w_vid:
                aux_row = cv2.resize(aux_row, (w_vid, preview_h))
            
            final_display = np.vstack((output_frame, aux_row))
            
            # Save to file
            if output_writer is not None and output_writer.isOpened():
                output_writer.write(final_display)
            
            # Encode and send
            ret_enc, buffer = cv2.imencode('.jpg', final_display, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ret_enc:
                await websocket.send(buffer.tobytes())
            
            await asyncio.sleep(0.001)
            
    except websockets.exceptions.ConnectionClosed:
        logging.info(f"Client {connection_id} disconnected")
    except Exception as e:
        logging.error(f"Error: {e}")
        traceback.print_exc()
    finally:
        duration = time.time() - start_time
        cap.release()
        active_connections.discard(websocket)
        
        if output_writer is not None:
            output_writer.release()
        
        logging.info(f"Processed {frame_count} frames in {duration:.2f}s ({frame_count/duration:.1f} FPS)")
        logging.info(f"Remaining clients: {len(active_connections)}")


async def start_server():
    """Start the WebSocket server."""
    print(f"\n{'='*70}")
    print(f"=== LiteDepth WebSocket Server Ready ===")
    print(f"=== Connect your client to: ws://localhost:{WEBSOCKET_PORT} ===")
    print(f"{'='*70}\n")
    
    server = await websockets.serve(
        video_stream_handler,
        "0.0.0.0",
        WEBSOCKET_PORT,
        ping_interval=20,
        ping_timeout=20,
        max_size=15*1024*1024
    )
    
    logging.info(f"Server running on 0.0.0.0:{WEBSOCKET_PORT}")
    await asyncio.Future()


# ==================== MAIN ====================

def select_video_file():
    """Let user select a video file."""
    global VIDEO_PATH
    
    script_dir = Path(__file__).parent
    videos_dir = script_dir / "Videos ( Input + Output)" / "Input"
    
    video_files = []
    
    # Check Videos folder
    if videos_dir.exists():
        video_files.extend(list(videos_dir.glob("*.mp4")))
        video_files.extend(list(videos_dir.glob("*.avi")))
        video_files.extend(list(videos_dir.glob("*.mov")))
    
    # Check current directory
    video_files.extend(list(script_dir.glob("*.mp4")))
    video_files.extend(list(script_dir.glob("*.avi")))
    
    if video_files:
        print("\n📹 Available video files:")
        for i, f in enumerate(video_files, 1):
            print(f"  {i}. {f.name}")
        
        while True:
            try:
                choice = input(f"\nSelect video (1-{len(video_files)}) or enter path: ").strip()
                
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(video_files):
                        VIDEO_PATH = str(video_files[idx])
                        break
                elif os.path.exists(choice):
                    VIDEO_PATH = choice
                    break
                
                print("Invalid selection. Try again.")
            except KeyboardInterrupt:
                print("\nExiting...")
                sys.exit(0)
    else:
        VIDEO_PATH = input("\nEnter full path to video file: ").strip()
    
    if not os.path.exists(VIDEO_PATH):
        logging.error(f"Video file not found: {VIDEO_PATH}")
        sys.exit(1)
    
    logging.info(f"Selected video: {VIDEO_PATH}")


def main():
    """Main entry point."""
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║     LiteDepth - Real-Time 3D Perception for Industrial Safety     ║
    ║             Caterpillar Tech Challenge 2025 Project               ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"🖥️  Device: {DEVICE.upper()}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"🔥 PyTorch: {torch.__version__}")
    
    # Load models
    print("\n📦 Loading models...")
    
    if not load_yolo_model():
        logging.error("Failed to load YOLO model. Exiting.")
        sys.exit(1)
    
    if not download_depth_model():
        logging.error("Failed to load depth model. Exiting.")
        sys.exit(1)
    
    # Select video
    print("\n📹 Select video file to process...")
    select_video_file()
    
    # Start server
    print(f"\n🚀 Starting WebSocket server on port {WEBSOCKET_PORT}...")
    print(f"📱 Open 'websocket.html' in your browser and connect to:")
    print(f"   ws://localhost:{WEBSOCKET_PORT}")
    
    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        logging.error(f"Server error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
