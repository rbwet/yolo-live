#!/usr/bin/env python3
"""
Real-time YOLOv8 ONNX inference pipeline — FPS / CoD tuned.

Pipeline: capture → preprocess → infer → postprocess → aimpoint

Usage:
    python yolo_live.py --model cod_enemy.onnx --data data.yaml   # class names from training YAML
    python yolo_live.py --aimbot --classes Enemy enemy_head       # override aimbotted classes
    python yolo_live.py --source dxcam                      # low-latency Windows capture
    python yolo_live.py --source mss                        # cross-platform fallback
    python yolo_live.py --input-size 320                    # faster inference (train matching size)
    python yolo_live.py --providers CUDAExecutionProvider   # GPU acceleration
    python yolo_live.py --aimbot                           # compute aimpoint + print coords

Dependencies:
    pip install opencv-python numpy onnxruntime mss dxcam
"""

import argparse
import ctypes
import math
import os
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort


def load_custom_classes(yaml_path: str = "data.yaml") -> list[str]:
    import yaml

    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    names = data["names"]  # list or {0: 'a', 1: 'b', ...} (Ultralytics)
    if isinstance(names, dict):
        return [names[i] for i in range(len(names))]
    return list(names)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "yolov8n.onnx"
INPUT_SIZE = 640           # model input dimension — lower = faster (320, 160)
CONF_THRESHOLD = 0.25      # filter detections below this confidence
NMS_IOU_THRESHOLD = 0.45   # NMS IoU suppression threshold

# Aimpoint smoothing — exponential moving average factor (0.0-1.0)
# Lower = smoother but slower to react; higher = snappier but jittery
AIM_SMOOTHING = 0.4

# Aimbotted class names (must match data.yaml "names" strings).
TARGET_CLASSES: list[str] = ["Enemy"]  # e.g. ["enemy_body", "enemy_head"]; [] = any class
AVOID_CLASSES: list[str] = []  # never aim at these


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str, providers: list[str] | None = None) -> ort.InferenceSession:
    """Load ONNX model and return an InferenceSession.

    Args:
        model_path: Path to the .onnx file.
        providers: Execution provider override, e.g. ["CUDAExecutionProvider", "CPUExecutionProvider"].
            If None, uses onnxruntime's default (CPU).
    """
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 2

    session = ort.InferenceSession(
        model_path,
        sess_options=opts,
        providers=providers or ["CPUExecutionProvider"],
    )
    return session


def get_model_info(session: ort.InferenceSession) -> dict:
    """Extract input / output metadata from the loaded session."""
    model_input = session.get_inputs()[0]
    return {
        "name": model_input.name,
        "shape": model_input.shape,          # [1, 3, H, W]
        "dtype": model_input.dtype,
    }


# ---------------------------------------------------------------------------
# Capture — screen or webcam
# ---------------------------------------------------------------------------

def capture_screen(region: tuple[int, int, int, int] | None = None) -> np.ndarray | None:
    """Capture a single frame from the primary monitor using mss.

    Args:
        region: Optional (left, top, width, height). When set, captures only that area.
            Useful for CoD windowed mode — avoids wasting cycles on HUD/taskbar.

    Returns BGR frame (uint8) or None on failure.
    """
    try:
        from mss import MSS          # type: ignore
    except ImportError:
        print("mss not installed — pip install mss")
        return None

    with MSS() as sct:
        monitor = region or sct.monitors[0]   # primary monitor unless region given
        img = sct.grab(monitor)
        frame = np.array(img)             # shape (H, W, 4), BGRA
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


def capture_dxcam(region: tuple[int, int, int, int] | None = None):
    """Return a DXCam screen-capture object for low-latency Windows capture.

    Returns the dxcam.Camera instance or None on failure. Callers must call
    .grab() each frame and pass through cv2.COLOR_BGRA2BGR conversion.
    """
    try:
        import dxcam               # type: ignore
    except ImportError:
        print("dxcam not installed — pip install dxcam")
        return None

    if region is not None:
        left, top, width, height = region
    else:
        left, top, width, height = 0, 0, 0, 0   # let DXCam auto-detect primary monitor

    cam = dxcam.create(
        output_bitrate=384_000,       # minimal bitrate — we only need raw pixels
        video_mode=True,              # capture desktop compositor (works in fullscreen)
        region=(left, top, width, height),
    )
    return cam


def _bgra_to_bgr(frame: np.ndarray) -> np.ndarray:
    """Convert BGRA numpy array to BGR."""
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


# ---------------------------------------------------------------------------
# Preprocess — letterbox resize + normalize to model input
# ---------------------------------------------------------------------------

def letterbox(
    img: np.ndarray,
    new_shape: tuple[int, int] = (INPUT_SIZE, INPUT_SIZE),
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize image with aspect-ratio-preserving letterbox padding.

    Returns:
        Letterboxed image (uint8), ratio used, (pad_w, pad_h).
    """
    shape = img.shape[:2][::-1]  # current shape [w, h]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
    new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))

    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw //= 2
    dh //= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=color)

    return img, r, (dw, dh)


def preprocess(frame: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Prepare a frame for model inference.

    Steps:
        1. Letterbox resize to INPUT_SIZE x INPUT_SIZE
        2. BGR → RGB
        3. HWC → CHW
        4. Normalize to [0, 1] float32
        5. Add batch dimension

    Returns:
        Preprocessed tensor, scale_ratio, padding_offset (dw, dh).
    """
    # Letterbox
    lb_img, ratio, pad = letterbox(frame, (INPUT_SIZE, INPUT_SIZE))

    # RGB + CHW + normalize
    img_rgb = lb_img[:, :, ::-1]                        # BGR → RGB
    img_chw = img_rgb.transpose(2, 0, 1)                # HWC → CHW
    img_tensor = img_chw.astype(np.float32) / 255.0     # [0,1]
    input_tensor = np.expand_dims(img_tensor, axis=0)   # [1, 3, H, W]

    return input_tensor, ratio, pad


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def infer(session: ort.InferenceSession, input_tensor: np.ndarray, input_name: str) -> np.ndarray:
    """Run a single forward pass and return the raw output tensor.

    YOLOv8 ONNX output shape: [1, 4 + num_classes, num_anchors] (e.g. [1, 84, 8400] for 80-class COCO).
    """
    outputs = session.run(None, {input_name: input_tensor})
    return np.squeeze(outputs[0], axis=0)  # [4 + num_classes, num_anchors]


# ---------------------------------------------------------------------------
# Postprocess — confidence filter + NMS → bounding boxes on original frame
# ---------------------------------------------------------------------------

COCLASS_NAMES = None   # lazily loaded


def _load_class_names(data_dir: str = "") -> list[str]:
    """Load COCO class names. Falls back to generic labels if coco.names not found."""
    global COCLASS_NAMES
    if COCLASS_NAMES is not None:
        return COCLASS_NAMES

    coco_path = Path(data_dir) / "coco.names" if data_dir else Path("coco.names")
    try:
        with open(coco_path) as f:
            COCLASS_NAMES = [line.strip() for line in f if line.strip()]
            return COCLASS_NAMES
    except FileNotFoundError:
        # 80-class COCO defaults
        COCLASS_NAMES = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
            "truck", "boat", "traffic light", "fire hydrant", "stop sign",
            "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
            "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
            "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
            "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
            "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv",
            "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush",
        ]
        return COCLASS_NAMES


def postprocess(
    output: np.ndarray,
    original_frame: np.ndarray,
    class_names: list[str],
    conf_thresh: float = CONF_THRESHOLD,
    iou_thresh: float = NMS_IOU_THRESHOLD,
    ratio: float = 1.0,
    pad: tuple[int, int] = (0, 0),
) -> tuple[np.ndarray, list[dict]]:
    """Apply confidence filtering and NMS to raw model output.

    Draws bounding boxes and labels on a copy of the original frame AND
    returns structured detection data for aimbot use.

    Args:
        output: Raw tensor from model [4 + num_classes, num_anchors] (num_classes is model-dependent).
        original_frame: Original captured frame (BGR).
        class_names: Per-class display names (same order as model training, e.g. from data.yaml).
        conf_thresh: Minimum confidence to keep a detection.
        iou_thresh: IoU threshold for NMS.
        ratio: Scale ratio from letterbox preprocessing.
        pad: Padding offset (dw, dh) from letterbox.

    Returns:
        (frame_with_boxes, list_of_detection_dicts). Each dict has keys:
            x1, y1, x2, y2, class_id, score
    """
    num_classes = output.shape[0] - 4
    output = output.T  # [num_anchors, 4 + num_classes]

    # --- Confidence filter ---
    box_confidence = np.max(output[:, 4:], axis=1, keepdims=True)   # per-anchor max class score [N, 1]
    class_scores = output[:, 4:]
    class_probs = class_scores * box_confidence  # combined score

    conf_mask = (class_probs >= conf_thresh).any(axis=1)             # keep anchors that have ANY class above threshold
    filtered_boxes = output[conf_mask][:, :4]                        # [N, 4] xywh normalized
    filtered_class_scores = output[conf_mask][:, 4:]                 # [N, num_classes]

    if len(filtered_boxes) == 0:
        return original_frame.copy(), []

    class_ids = np.argmax(filtered_class_scores, axis=1)
    confidences = (filtered_class_scores * box_confidence[conf_mask])[np.arange(len(class_ids)), class_ids]

    # --- NMS ---
    boxes_xyxy_int = _xywh_to_xyxy(filtered_boxes).astype(np.int32).tolist()
    indices_list = cv2.dnn.NMSBoxes(
        boxes_xyxy_int,
        confidences.tolist(),
        conf_thresh,
        iou_thresh,
    )

    # --- Build detections list and draw on frame ---
    detections: list[dict] = []
    h, w = original_frame.shape[:2]

    if len(indices_list) > 0:
        for idx in indices_list.flatten():
            box = filtered_boxes[idx].astype(np.float32)             # xywh normalized [4]
            cid = int(class_ids[idx])
            score = float(confidences[idx])

            x1f, y1f, x2f, y2f = _xywh_to_xyxy(box.reshape(1, 4))[0]
            x1 = max(0,         min(w, int((x1f - pad[0]) / ratio)))
            y1 = max(0,         min(h, int((y1f - pad[1]) / ratio)))
            x2 = max(0,         min(w, int((x2f - pad[0]) / ratio)))
            y2 = max(0,         min(h, int((y2f - pad[1]) / ratio)))

            det = {
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "class_id": cid,
                "score": score,
            }
            detections.append(det)

    frame_out = original_frame.copy()
    for det in detections:
        cid = int(det["class_id"])
        cname = class_names[cid % len(class_names)] if class_names else f"class_{cid}"
        label = f"{cname} {det['score']:.2f}"
        _draw_box(frame_out, (det["x1"], det["y1"], det["x2"], det["y2"]), label, det["class_id"])

    return frame_out, detections


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert [x_center, y_center, width, height] to [x1, y1, x2, y2]."""
    xc, yc, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.column_stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])


def _draw_box(frame: np.ndarray, box: tuple[int, int, int, int], label: str, class_id: int):
    """Draw a single bounding box with label on the frame."""
    color = _class_color(class_id)
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Label background
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def _class_color(class_id: int) -> tuple[int, int, int]:
    """Deterministic BGR color per class."""
    np.random.seed(class_id)
    rgb = np.random.randint(0, 255, size=3).tolist()
    return (rgb[2], rgb[1], rgb[0])  # RGB → BGR


# ---------------------------------------------------------------------------
# Virtual Controller — vJoy analog stick injection (Windows only)
# ---------------------------------------------------------------------------

@dataclass
class StickState:
    """Smoothed state for virtual right-stick injection."""
    smooth_x: float = 0.0       # EMA-smoothed horizontal aim offset [-1, 1]
    smooth_y: float = 0.0       # EMA-smoothed vertical aim offset   [-1, 1]


class VirtualController:
    """Inject analog stick axes via vJoy (Windows).

    Falls back to no-op if not on Windows or vjoy.dll is missing — useful for testing.
    Axis range: INT16 (-32768 .. +32767), mapped from normalized [-1, 1].
    """

    def __init__(self):
        self._vjoy = None
        self._vid = 1           # vJoy instance ID
        self.state = StickState()

        if sys.platform != "win32":
            print("  [VirtualController] Not on Windows — using no-op backend.")
            return

        try:
            dll_path = os.environ.get("VJOY_DLL", "")
            lib_name = dll_path if dll_path else "vJoy"
            self._vjoy = ctypes.WinDLL(lib_name)   # type: ignore[attr-defined]

            # vJoy constants (from VJD_SDK header):
            # 0x2304 = AXIS.RX, 0x2305 = AXIS.RY
            if not getattr(self._vjoy, "InitVJD")(self._vid):
                print(f"  [VirtualController] InitVJD({self._vid}) failed — using no-op.")
                self._vjoy = None

        except OSError as e:
            print(f"  [VirtualController] vJoy DLL not found ({e}). Install vJoy or set VJOY_DLL env var.")

    def _val(self, norm: float) -> int:
        """Convert normalized [-1, 1] to INT16 range."""
        clamped = max(-1.0, min(1.0, norm))
        return int(clamped * 32767)

    # vJoy axis constants (from VJD_SDK.h): AXIS.RX=0x2304, AXIS.RY=0x2305
    _RX = ctypes.c_int(0x2304)
    _RY = ctypes.c_int(0x2305)

    def inject_axis(self, x_norm: float, y_norm: float):
        """Set right-stick X/Y axes on the virtual controller."""
        if self._vjoy is None:
            return  # no-op fallback

        self._vjoy.SetVJDAxisParam(
            ctypes.c_ulong(self._vid), self._RX, ctypes.c_long(self._val(x_norm))
        )
        self._vjoy.SetVJDAxisParam(
            ctypes.c_ulong(self._vid), self._RY, ctypes.c_long(self._val(y_norm))
        )

    def release_stick(self):
        """Center the stick (return to neutral)."""
        if self._vjoy is None:
            return
        self.inject_axis(0.0, 0.0)


# ---------------------------------------------------------------------------
# Aimpoint — compute analog-stick deltas from detections
# ---------------------------------------------------------------------------

def select_target(
    detections: list[dict],
    frame_h: int,
    frame_w: int,
    class_names: list[str],
    target_classes: list[str],
) -> Optional[dict]:
    """Pick the best detection to aim at.

    Priority rules for CoD:
      1. Must be in target_classes by name (not in AVOID_CLASSES), or target_classes [] = any.
      2. Prefer closest-to-center targets (least stick travel needed).
      3. If multiple equally close, prefer highest confidence.

    Returns None if no valid target found.
    """
    candidates = []

    for det in detections:
        cid = int(det["class_id"])

        # Handle color-tracking class IDs directly (-99=red enemy, -98=yellow teammate)
        if cid == -99:  # red health bar → always aimable
            candidates.append(det)
            continue
        elif cid == -98:  # yellow health bar → never aim at teammates
            continue

        if class_names:
            cname = class_names[cid % len(class_names)]
            if cname in AVOID_CLASSES:
                continue
            if target_classes and cname not in target_classes:
                continue

        candidates.append(det)

    if not candidates:
        return None

    # Sort by distance to screen center, then confidence descending
    cx = frame_w / 2
    cy = frame_h / 2
    for c in candidates:
        tcx = (c["x1"] + c["x2"]) / 2
        tcy = (c["y1"] + c["y2"]) / 2
        c["_dist_to_center"] = math.hypot(tcx - cx, tcy - cy)

    candidates.sort(key=lambda d: (d["_dist_to_center"], -d["score"]))
    return candidates[0]


def compute_stick_input(
    target: dict,
    frame_w: int,
    frame_h: int,
    controller: VirtualController,
    smoothing: float = AIM_SMOOTHING,
) -> tuple[float, float]:
    """Convert a detected bounding box into analog stick deltas [-1..1].

    The error is proportional to how far the target center is from screen center.
    Full deflection when target is at screen edge; zero when crosshair-on-target.
    This works with CoD's in-game look sensitivity + aim assist naturally.
    """
    # Target box center as fraction of frame [-0.5, 0.5] → normalize to [-1, 1]
    tcx = (target["x1"] + target["x2"]) / 2
    tcy = (target["y1"] + target["y2"]) / 2

    error_x = ((tcx - frame_w / 2) / (frame_w / 2))   # [-1, 1] right is positive
    error_y = ((tcy - frame_h / 2) / (frame_h / 2))   # [-1, 1] down is positive

    # EMA smoothing: blend new reading with previous state
    controller.state.smooth_x = (smoothing * error_x) + ((1.0 - smoothing) * controller.state.smooth_x)
    controller.state.smooth_y = (smoothing * error_y) + ((1.0 - smoothing) * controller.state.smooth_y)

    # Deadzone: don't inject tiny movements that fight aim assist
    DEADZONE = 0.02
    if abs(controller.state.smooth_x) < DEADZONE:
        controller.state.smooth_x = 0.0
    if abs(controller.state.smooth_y) < DEADZONE:
        controller.state.smooth_y = 0.0

    # Scale output — full stick deflection can be too aggressive; dial down for natural feel
    SCALE_FACTOR = 1.2   # >1 means we push harder than raw error (aggressive tracking)
    inject_x = controller.state.smooth_x * SCALE_FACTOR
    inject_y = controller.state.smooth_y * SCALE_FACTOR

    return inject_x, inject_y


# ---------------------------------------------------------------------------
# Color Tracking — detect enemy health bars (red FF0000) / teammates (FFFF00)
# ---------------------------------------------------------------------------

def track_colors(
    frame: np.ndarray,
    min_area: int = 120,       # minimum contour area to avoid noise and UI artifacts.
) -> list[dict]:
    """Detect enemy health bars via HSV color masking.

    Red (#FF0000 / BGR=(0,0,255)) marks enemies; yellow (#FFFF00 / BGR=(0,255,255)) marks teammates.
    Red wraps around the HSV hue boundary (H≈0° and H≈180°), so we mask both halves and OR them.
    Red detections become enemy targets (aimable); yellow detections are marked as teammates so
    select_target skips them.

    Args:
        frame: Original captured frame in BGR.
        min_area: Minimum contour pixel area; filters out noise and UI artifacts.

    Returns:
        List of detection dicts compatible with select_target(). Each has keys:
            x1, y1, x2, y2, class_id (-99=red enemy / -98=yellow teammate), score (0-1)
    """
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red hue wraps in OpenCV's [0, 179] hue space — mask both ends and combine.
    red_lower_1 = np.array([0, 120, 100])
    red_upper_1 = np.array([10, 255, 255])
    red_lower_2 = np.array([170, 120, 100])
    red_upper_2 = np.array([179, 255, 255])
    mask_red = cv2.inRange(frame_hsv, red_lower_1, red_upper_1) | cv2.inRange(
        frame_hsv, red_lower_2, red_upper_2
    )

    # BGR(0,255,255) → HSV H≈60° (OpenCV /2 ≈30). Range 18-34 covers yellow region.
    yellow_lower = np.array([18, 100, 100])
    yellow_upper = np.array([34, 255, 255])
    mask_yel = cv2.inRange(frame_hsv, yellow_lower, yellow_upper)

    results: list[dict] = []

    for mask, class_id in [(mask_red, -99), (mask_yel, -98)]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            x1, y1, bw, bh = cv2.boundingRect(cnt)
            results.append({
                "x1": x1, "y1": y1,
                "x2": x1 + bw, "y2": y1 + bh,
                "class_id": class_id,
                "score": 0.95 if class_id == -99 else 0.80,   # color detections are high confidence
            })

    return results


# ---------------------------------------------------------------------------
# Display — render frame with FPS overlay + aimpoint indicator
# ---------------------------------------------------------------------------

def display(frame: np.ndarray, fps: float):
    """Show the processed frame in an OpenCV window with FPS counter."""
    h = frame.shape[0]
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
    )
    cv2.imshow("YOLOv8 Live Detection", frame)


def draw_target_indicator(frame: np.ndarray, target: dict):
    """Draw a crosshair at the center of the selected aimpoint."""
    cx = (target["x1"] + target["x2"]) // 2
    cy = (target["y1"] + target["y2"]) // 2

    cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
    size = 8
    thickness = 1
    # Crosshair arms
    cv2.line(frame, (cx - size, cy), (cx + size, cy), (0, 255, 0), thickness)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), (0, 255, 0), thickness)


# ---------------------------------------------------------------------------
# Main entry point — parse args and run inference loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Real-time YOLOv8 ONNX inference")
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Path to ONNX model file (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--source", choices=["screen", "dxcam", "webcam"], default="screen",
        help="Capture source (default: screen)",
    )
    parser.add_argument("--device", type=int, default=0, help="Webcam device ID")

    # Region: capture only part of the screen. Format: left,top,width,height
    parser.add_argument(
        "--region", nargs=4, metavar=("L", "T", "W", "H"), type=int, default=None,
        help="Capture region (left top width height). Skips HUD/taskbar in CoD windowed mode.",
    )

    # Aimbot flags
    parser.add_argument(
        "--aimbot", action="store_true",
        help="Enable aimbot: select target + inject virtual controller stick movement via vJoy",
    )
    parser.add_argument(
        "--smoothing", type=float, default=AIM_SMOOTHING,
        help=f"EMA smoothing factor for aimpoint (default: {AIM_SMOOTHING})",
    )

    # Tuning params
    parser.add_argument("--input-size", type=int, default=None, dest="input_size")
    parser.add_argument(
        "--conf", type=float, default=CONF_THRESHOLD,
        help=f"Confidence threshold (default: {CONF_THRESHOLD})",
    )
    parser.add_argument(
        "--iou", type=float, default=NMS_IOU_THRESHOLD,
        help=f"NMS IoU threshold (default: {NMS_IOU_THRESHOLD})",
    )
    parser.add_argument(
        "--providers", nargs="+", default=None,
        help='Execution providers, e.g. CUDAExecutionProvider CPUExecutionProvider',
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data.yaml",
        help="Dataset YAML (Ultralytics) with a 'names' field for on-screen class labels; "
        "if missing, falls back to ./coco.names or built-in COCO names",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        metavar="NAME",
        help="Aim at these class name(s), matching data.yaml. Omit the flag to use script "
        "TARGET_CLASSES; use --classes with no names to allow any class",
    )
    args = parser.parse_args()

    # Override global INPUT_SIZE if specified on CLI (must match trained model)
    if args.input_size is not None:
        globals()["INPUT_SIZE"] = args.input_size

    region = tuple(args.region) if args.region else None   # type: ignore[assignment]

    # --- Load model ---
    print(f"Loading model: {args.model}")
    session = load_model(args.model, args.providers)
    info = get_model_info(session)
    input_name = info["name"]
    active_providers = session.get_providers()
    print(f"  Input : {info['shape']}, dtype={info['dtype']}")
    print(f"  Providers: {active_providers}")

    data_yaml = Path(args.data)
    if data_yaml.is_file():
        class_names: list[str] = load_custom_classes(str(data_yaml))
        print(f"  Class names: {len(class_names)} from {data_yaml}")
    else:
        class_names = _load_class_names()
        print(f"  {data_yaml} not found — labels from coco.names or built-in COCO ({len(class_names)} names)")

    if args.classes is None:
        target_classes = list(TARGET_CLASSES)
    else:
        target_classes = list(args.classes)
    if not target_classes:
        print("  Aim classes: (any / no name filter)")
    else:
        print(f"  Aim classes: {target_classes}")

    # --- Setup capture ---
    cap = None
    dxcam_obj = None
    if args.source == "webcam":
        cap = cv2.VideoCapture(args.device)
        if not cap.isOpened():
            print(f"Error: could not open webcam device {args.device}")
            return
        print(f"Webcam opened (device={args.device})")

    elif args.source == "dxcam":
        dxcam_obj = capture_dxcam(region=region)
        if dxcam_obj is None:
            print("Error: DXCam creation failed — falling back to mss.")
            # fall through to screen mode by clearing source flag
            args.source = "screen"

    elif region is not None:
        l, t, w, h = region
        print(f"Cropping capture to region: ({l},{t}) {w}x{h}")

    # --- Setup aimbot (virtual controller) ---
    controller = VirtualController() if args.aimbot else None
    if controller and not controller._vjoy:
        print("  [Aimbot] Running in no-op mode (not on Windows / vJoy unavailable). Stick values printed to console.")

    # Key state tracking for trigger-based aimbot activation
    last_target = None   # remember lock across frames even when detection flickers

    print("\nPress 'q' to quit.  Press SPACE to toggle aimbot ON/OFF (when --aimbot enabled).")

    # --- Main loop ---
    frame_count = 0
    start_time = time.monotonic()
    aimbot_active = args.aimbot   # starts on if flag passed, user can space-toggle

    try:
        while True:
            # Capture
            if dxcam_obj is not None and args.source == "dxcam":
                raw_frame = dxcam_obj.grab()
                frame = _bgra_to_bgr(np.asarray(raw_frame))  # type: ignore[arg-type]
                del raw_frame   # free DXCam buffer ASAP for memory
            elif args.source == "screen" or (args.source != "dxcam"):
                frame = capture_screen(region=region)
                if frame is None:
                    print("Screen capture failed, retrying...")
                    continue
            else:  # webcam
                ret, frame = cap.read()
                if not ret:
                    print("Webcam read failed")
                    break

            h, w = frame.shape[:2]

            # Preprocess
            input_tensor, ratio, pad = preprocess(frame)

            # Inference
            output = infer(session, input_tensor, input_name)

            # Postprocess — now returns (frame_with_boxes, detections_list)
            result_frame, detections = postprocess(
                output, frame, class_names,
                conf_thresh=args.conf,
                iou_thresh=args.iou,
                ratio=ratio,
                pad=pad,
            )

            # Color tracking: detect red enemy / yellow teammate health bars independently of YOLO
            color_dets = track_colors(frame) if controller is not None and aimbot_active else []
            all_detections = detections + color_dets

            # --- Aimbot: select target + inject stick movement ---
            if controller is not None and aimbot_active and all_detections:
                target = select_target(all_detections, h, w, class_names, target_classes)
                if target is not None:
                    last_target = target
                    ix, iy = compute_stick_input(target, w, h, controller, smoothing=args.smoothing)   # type: ignore[misc]
                    controller.inject_axis(ix, iy)

                    # Draw lock-on indicator on the display frame
                    draw_target_indicator(result_frame, target)

                    if not controller._vjoy:  # no-op mode — print for debugging
                        cv2.putText(
                            result_frame, f"STICK [{ix:+.3f} {iy:+.3f}]", (10, h - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1,
                        )

                elif last_target is not None:
                    # Detection flickered this frame — keep using previous lock briefly
                    ix = controller.state.smooth_x * 1.2   # type: ignore[misc]
                    iy = controller.state.smooth_y * 1.2    # type: ignore[misc]
                    _DZ = 0.02    # inline deadzone (same value as compute_stick_input)
                    if abs(ix) > _DZ or abs(iy) > _DZ:
                        controller.inject_axis(ix, iy)

            elif controller is not None and aimbot_active:
                # No detections this frame — release stick to neutral
                controller.release_stick()
                last_target = None

            # FPS calculation
            frame_count += 1
            elapsed = time.monotonic() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            # Show aimbot status on overlay (only when --aimbot passed)
            if controller is not None:
                state_text = "AIMBOT ON" if aimbot_active else "AIMBOT OFF [space to enable]"
                cv2.putText(
                    result_frame, state_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0) if aimbot_active else (0, 0, 255), 1,
                )

            # Display
            display(result_frame, fps)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" ") and controller is not None:
                aimbot_active = not aimbot_active   # toggle with spacebar
                print(f" Aimbot {'enabled' if aimbot_active else 'disabled'}")

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        if cap:
            cap.release()
        controller.release_stick()  # type: ignore[union-attr]

    print(f"\nDone. Processed {frame_count} frames in {elapsed:.1f}s "
          f"({fps:.1f} FPS avg)")


if __name__ == "__main__":
    main()
