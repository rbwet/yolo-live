<div align="center">

# yolo-live

**Real-time YOLOv8 ONNX detection pipeline, tuned for 60+ FPS gameplay.**

Capture → preprocess → infer → postprocess → aimpoint, in one tight loop.

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.16%2B-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Ultralytics](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-46aef7)](https://github.com/astral-sh/ruff)

</div>

---

## Why this exists

Most "real-time YOLO" demos drop frames the moment you hand them a 1080p source.
This repo is a **minimal, end-to-end pipeline** that:

- Captures the screen (or a webcam) with low-latency backends (`mss`, `dxcam`).
- Runs a quantization-friendly **ONNX** graph through `onnxruntime` — CPU or CUDA.
- Letterboxes, infers, and applies vectorized NMS in NumPy — **no torch at runtime**.
- Picks an aimpoint, smooths it with an EMA, and renders an overlay you can tune live.
- Comes with a complete **train-your-own-class** workflow (extract → autolabel → fine-tune → export).

Built originally to detect enemy players in *Call of Duty* gameplay clips — the same
pipeline works for any single-class detector you want to ship in real time.

---

## Pipeline

```
┌──────────┐   ┌────────────┐   ┌──────────┐   ┌────────────┐   ┌──────────┐
│  capture │──▶│ letterbox  │──▶│  ONNX    │──▶│  NMS +     │──▶│ aimpoint │
│ mss/dxcam│   │ + normalize│   │ inference│   │ class filt │   │  + EMA   │
└──────────┘   └────────────┘   └──────────┘   └────────────┘   └──────────┘
     │                                                                │
     └─────────────── per-frame BGR ──────────────────────────── overlay + log
```

Every stage lives in `yolo_live.py` as a small, pure function — easy to swap in a
TensorRT session, a pose head, or a tracker without rewriting the loop.

---

## Quick start

```bash
git clone https://github.com/rbwet/yolo-live.git
cd yolo-live
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-yolo.txt

# Grab a pretrained COCO model and convert to ONNX
python export_yolo_to_onnx.py --weights yolov8n.pt

# Run on your screen
python yolo_live.py --model yolov8n.onnx
```

Hot keys in the overlay window: `q` quits, `s` saves the current frame.

### Common variations

```bash
# CUDA inference
python yolo_live.py --model yolov8n.onnx --providers CUDAExecutionProvider

# Tighter region (skip HUD / taskbar) + lower input size = much higher FPS
python yolo_live.py --region 0 100 1920 880 --input-size 320

# Webcam source
python yolo_live.py --source webcam --device 0

# Use class names from a custom data.yaml
python yolo_live.py --model cod_enemy.onnx --data data.yaml
```

---

## Train your own detector

Everything you need to fine-tune YOLOv8 on your own footage lives in `tools/`.

### 1. Drop clips into `clips/`

Any mp4/mov/mkv. Symlinks work too:

```bash
ln -s "/path/to/match.mp4" clips/match.mp4
```

### 2. Extract frames

```bash
python tools/extract_frames.py --fps 2 --dedup 4.0
```

Walks `clips/`, samples at 2 fps, drops near-duplicate frames into `dataset/images/`.

### 3. Bootstrap labels with a COCO model

```bash
python tools/autolabel_enemies.py \
    --model yolov8n.onnx \
    --images dataset/images \
    --labels dataset/labels \
    --conf 0.30
```

Every detected `person` becomes class `0 = Enemy`. **You still need to clean these up**
in [Roboflow](https://roboflow.com/), Label Studio, or CVAT — teammates and bots get
boxed too.

### 4. Train (Colab notebook)

Open [`tools/train_bo7_colab.ipynb`](tools/train_bo7_colab.ipynb) on a free T4 and run
it top to bottom. It uploads your dataset, fine-tunes `yolov8n`, and downloads
`best.pt` when done.

### 5. Export and run

```bash
python export_yolo_to_onnx.py --weights runs/detect/train/weights/best.pt -o cod_enemy.onnx
python yolo_live.py --model cod_enemy.onnx --data data.yaml
```

---

## Project layout

```
yolo-live/
├── yolo_live.py             # real-time inference loop (capture → overlay)
├── export_yolo_to_onnx.py   # YOLOv8 .pt → .onnx
├── data.yaml                # class names + dataset root
├── requirements-yolo.txt    # runtime deps
├── tools/
│   ├── extract_frames.py        # video → dedup'd JPEGs
│   ├── autolabel_enemies.py     # COCO model → bootstrap YOLO labels
│   └── train_bo7_colab.ipynb    # fine-tune YOLOv8n on Colab
├── dataset/                 # images/ + labels/ (gitignored)
└── clips/                   # raw video footage (gitignored)
```

---

## Tuning knobs

| Flag | Default | What it does |
|------|---------|--------------|
| `--input-size` | model native | Smaller = faster. Train and infer at the same size. |
| `--conf` | `0.25` | Drop detections below this confidence. |
| `--iou` | `0.45` | NMS overlap threshold. |
| `--smoothing` | `0.4` | EMA factor for the aimpoint (0 = stuck, 1 = jittery). |
| `--region L T W H` | full screen | Crop the capture area; huge FPS win in windowed mode. |
| `--providers` | CPU | e.g. `CUDAExecutionProvider`, `CoreMLExecutionProvider`. |
| `--classes` | from yaml | Override the names targeted by the aimpoint logic. |

Run `python yolo_live.py --help` for the full list.

---

## Performance notes

- **Letterbox once, in numpy** — no PIL, no torch tensor copies.
- `onnxruntime` is configured with `intra_op_num_threads=2` and a single inter-op
  thread. On a quiet CPU this keeps the inference time variance tight.
- The screen-capture path uses `dxcam` on Windows and `mss` everywhere else; both
  hand back zero-copy BGRA buffers that we convert in-place.
- For aim-stick injection on Windows, the optional `--aimbot` path talks to
  [vJoy](http://vjoystick.sourceforge.net/) over the standard HID API.

---

## Hardware requirements

### PC gaming — no extra hardware needed

For **PC games**, this runs entirely on your machine. No capture cards, no external gear:

```
Your GPU renders the game → dxcam/mss grabs frames in-memory → YOLOv8 processes them → overlay drawn on top
```

All happening locally with zero encoding/decoding latency — that's why it hits 60+ FPS. The only requirement is a GPU capable of pushing ONNX inference at speed (basically any modern card). Just `pip install` and run alongside your game.

### Console gaming — capture card required

For **PS5 / Xbox**, you'll need an external capture card since you can't run Python on the console itself:

```
Console → HDMI out → Capture card → USB into PC → CV pipeline runs on PC
```

**Trade-offs:** adds latency, drops quality (compressed video stream instead of raw frame buffers), and costs $100–200 for a decent card. Also slower overall since you're dealing with encoded streams rather than direct memory access.

---

## Disclaimer

This code is for **research, education, and offline analysis of gameplay you
recorded yourself**. Using detection-driven input injection in online competitive
games will get your account banned and is against the ToS of every major title.
Don't be that person.

---

## License

[MIT](LICENSE) — do whatever you want, just don't blame me.
