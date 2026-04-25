#!/usr/bin/env python3
"""Bootstrap YOLO labels by pre-boxing every 'person' as class 'Enemy'.

Runs a COCO-pretrained YOLOv8 ONNX model over every image in --images and
writes one YOLO-format .txt label file per image to --labels. Every
detected 'person' (COCO id 0) becomes class 0 = 'Enemy' in the output.

YOU WILL STILL NEED TO REVIEW THESE. Teammates, bots, etc. will be boxed too;
remove / fix them in Roboflow (or Label Studio / CVAT) before training.

Usage:
    python tools/autolabel_enemies.py \\
        --model yolov8n.onnx \\
        --images dataset/images \\
        --labels dataset/labels \\
        --conf 0.30

Requires a COCO-class YOLOv8 ONNX model in the project root (e.g. yolov8n.onnx).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


PERSON_COCO_ID = 0
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def letterbox(
    img: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    shape = img.shape[:2][::-1]  # w, h
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
    new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw //= 2
    dh //= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def preprocess(frame: np.ndarray, size: int):
    lb, ratio, pad = letterbox(frame, (size, size))
    rgb = lb[:, :, ::-1]
    chw = rgb.transpose(2, 0, 1)
    tensor = np.expand_dims(chw.astype(np.float32) / 255.0, 0)
    return tensor, ratio, pad


def iou_xyxy(boxes: np.ndarray, pick: np.ndarray) -> np.ndarray:
    x1 = np.maximum(boxes[:, 0], pick[0])
    y1 = np.maximum(boxes[:, 1], pick[1])
    x2 = np.minimum(boxes[:, 2], pick[2])
    y2 = np.minimum(boxes[:, 3], pick[3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area_p = (pick[2] - pick[0]) * (pick[3] - pick[1])
    return inter / (area_b + area_p - inter + 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    idxs = scores.argsort()[::-1]
    keep: list[int] = []
    while len(idxs):
        i = int(idxs[0])
        keep.append(i)
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        ious = iou_xyxy(boxes[rest], boxes[i])
        idxs = rest[ious < iou_thresh]
    return keep


def run_one(
    session: ort.InferenceSession,
    input_name: str,
    img: np.ndarray,
    size: int,
    conf_thresh: float,
    iou_thresh: float,
) -> list[tuple[float, float, float, float, float]]:
    """Return list of (x1,y1,x2,y2,score) person boxes in original image coords."""
    h, w = img.shape[:2]
    tensor, ratio, (dw, dh) = preprocess(img, size)
    out = session.run(None, {input_name: tensor})[0]  # [1, 4+nc, N]
    out = np.squeeze(out, 0)

    nc = out.shape[0] - 4
    if nc < PERSON_COCO_ID + 1:
        return []  # model doesn't have 'person' class (probably not COCO)

    out = out.T  # [N, 4+nc]
    cls_scores = out[:, 4:]
    cls_ids = cls_scores.argmax(1)
    cls_conf = cls_scores.max(1)

    mask = (cls_ids == PERSON_COCO_ID) & (cls_conf >= conf_thresh)
    if not np.any(mask):
        return []

    xywh = out[mask, :4]
    scores = cls_conf[mask]

    x = xywh[:, 0]
    y = xywh[:, 1]
    bw = xywh[:, 2]
    bh = xywh[:, 3]
    x1 = x - bw / 2
    y1 = y - bh / 2
    x2 = x + bw / 2
    y2 = y + bh / 2

    x1 = (x1 - dw) / ratio
    y1 = (y1 - dh) / ratio
    x2 = (x2 - dw) / ratio
    y2 = (y2 - dh) / ratio

    x1 = np.clip(x1, 0, w - 1)
    y1 = np.clip(y1, 0, h - 1)
    x2 = np.clip(x2, 0, w - 1)
    y2 = np.clip(y2, 0, h - 1)

    boxes = np.stack([x1, y1, x2, y2], axis=1)
    keep = nms(boxes, scores, iou_thresh)
    return [(float(boxes[i, 0]), float(boxes[i, 1]), float(boxes[i, 2]), float(boxes[i, 3]), float(scores[i])) for i in keep]


def xyxy_to_yolo(box: tuple[float, float, float, float], w: int, h: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return cx, cy, bw, bh


def main():
    parser = argparse.ArgumentParser(description="Bootstrap YOLO labels: person -> Enemy (class 0)")
    parser.add_argument("--model", type=str, default="yolov8n.onnx", help="COCO-pretrained YOLOv8 ONNX")
    parser.add_argument("--images", type=Path, default=Path("dataset/images"))
    parser.add_argument("--labels", type=Path, default=Path("dataset/labels"))
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument(
        "--providers",
        nargs="+",
        default=None,
        help="ONNX Runtime providers (e.g. CUDAExecutionProvider CPUExecutionProvider)",
    )
    args = parser.parse_args()

    if not Path(args.model).is_file():
        print(f"Model not found: {args.model}")
        sys.exit(1)
    if not args.images.is_dir():
        print(f"Images dir not found: {args.images}")
        sys.exit(1)

    args.labels.mkdir(parents=True, exist_ok=True)

    opts = ort.SessionOptions()
    session = ort.InferenceSession(
        args.model, sess_options=opts, providers=args.providers or ["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    print(f"Model: {args.model}  providers={session.get_providers()}")

    images = [p for p in sorted(args.images.iterdir()) if p.suffix.lower() in IMG_EXTS]
    if not images:
        print(f"No images in {args.images}")
        sys.exit(1)

    print(f"Auto-labeling {len(images)} images -> {args.labels}")
    boxed = 0
    empty = 0
    for i, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [skip] unreadable: {img_path}")
            continue
        h, w = img.shape[:2]
        dets = run_one(session, input_name, img, args.size, args.conf, args.iou)

        label_path = args.labels / (img_path.stem + ".txt")
        if not dets:
            label_path.write_text("")  # empty label file = "no enemies" for YOLO
            empty += 1
        else:
            lines = []
            for (x1, y1, x2, y2, score) in dets:
                cx, cy, bw, bh = xyxy_to_yolo((x1, y1, x2, y2), w, h)
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            label_path.write_text("\n".join(lines) + "\n")
            boxed += 1

        if i % 50 == 0 or i == len(images):
            print(f"  {i}/{len(images)}  boxed={boxed}  empty={empty}")

    print(f"\nDone. Labels written to {args.labels}  (with-boxes: {boxed}, empty: {empty})")
    print("Next: upload dataset/images + dataset/labels + data.yaml to Roboflow, review, and train.")


if __name__ == "__main__":
    main()
