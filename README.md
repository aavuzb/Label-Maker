<div align="center">

# LabelMaker

**A desktop dataset preparation tool for computer vision.**

Label images for **Classification**, **Detection**, and **Segmentation** — by hand, or automatically with a model you already have — and export a training-ready dataset when you're done.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![PySide6](https://img.shields.io/badge/UI-PySide6%20(Qt)-41cd52)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20%7C%20macOS-lightgrey)

</div>

---

## Contents

- [What is LabelMaker?](#what-is-labelmaker)
- [Features](#features)
- [Screenshots](#screenshots)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Manual Labeling](#manual-labeling)
- [Auto-Labeling](#auto-labeling)
- [Exporting a Dataset](#exporting-a-dataset)
- [Keyboard Shortcuts](#keyboard-shortcuts)
- [Project Structure](#project-structure)
- [License](#license)

---

## What is LabelMaker?

LabelMaker is a self-contained desktop app for preparing image datasets before training a computer vision model. It covers the three most common labeling tasks in one tool:

| Task | What you produce |
|---|---|
| **Classification** | One class per whole image (*"this photo is a cat"*) |
| **Detection** | A bounding box around every object in an image |
| **Segmentation** | A precise polygon outline around every object |

Every project can be labeled **manually** — by clicking, drawing, or tracing directly in the app — **automatically**, by running an existing trained model over your images and reviewing/keeping its predictions, or a mix of both (auto-label first, then manually fix up the rest). When you're done, export straight into a folder structure standard training frameworks already expect.

Your original image files are never touched by labeling itself — everything is tracked in a project file next to your images, and nothing autosaves *into* your source folders except Classification's optional "Move" mode, which you turn on deliberately.

## Features

- **Three labeling workflows** — Classification, Detection, and Segmentation, each with a purpose-built canvas and its own keyboard-driven workflow.
- **Manual labeling** — click-to-assign for classification, click-and-drag boxes for detection, click-to-trace polygons for segmentation. Edit, reassign, or delete any label at any time.
- **Auto-labeling** — point the app at a model you've already trained and let it label your images for you, with a confidence threshold and a live progress report.
- **Broad model format support** — ONNX, Darknet, Caffe, TensorFlow frozen graphs, PyTorch TorchScript, and raw Ultralytics YOLO checkpoints (detect, classify, *and* segment) — see [Auto-Labeling](#auto-labeling) below.
- **GPU acceleration** — auto-labeling runs on your NVIDIA GPU when available (CUDA, via onnxruntime-gpu or PyTorch), with automatic CPU fallback and live GPU status in the settings dialog.
- **Dataset export** — every task exports to the file layouts real training tools expect (see [Exporting a Dataset](#exporting-a-dataset)), with a configurable train/val/test split.
- **Autosave, always** — every change (a label, a box, a new class) is written to disk immediately. No Save button, nothing to lose if the app closes unexpectedly.
- **Non-destructive by default** — labeling reads your images in place; nothing is copied, moved, or modified unless you explicitly export or opt into Classification's Move mode.

## Screenshots

### Getting started

<table>
<tr>
<td><img src="screenshots/New%20Classification%20Project.png" alt="Creating a new Classification project"></td>
<td><img src="screenshots/New%20Detection%20Project.png" alt="Creating a new Detection project"></td>
</tr>
<tr>
<td align="center"><i>Creating a new Classification project</i></td>
<td align="center"><i>Creating a new Detection project</i></td>
</tr>
<tr>
<td><img src="screenshots/New%20Segmentation%20Project.png" alt="Creating a new Segmentation project"></td>
<td><img src="screenshots/Open%20Existed%20Projects.png" alt="Reopening an existing project"></td>
</tr>
<tr>
<td align="center"><i>Creating a new Segmentation project</i></td>
<td align="center"><i>Reopening any project you've worked on before</i></td>
</tr>
</table>

### Classification

| | |
|---|---|
| ![Classification workspace after auto-labeling](screenshots/Classification%20dataset%20after%20Auto-Labeling.png) | ![Editing an image's class](screenshots/Edit%20class%20of%20image.png) |
| *Labeled images, with the assigned class shown on the preview* | *Reassigning or removing a label from the image grid* |
| ![Auto-Label Settings for Classification](screenshots/Auto-Label%20Settings%20-%20Classification.png) | ![Auto-labeling progress for Classification](screenshots/Auto-Label%20Report%20-%20Classification.png) |
| *Choosing a model and options before auto-labeling* | *Live progress while auto-labeling runs* |
| ![Export Classification dataset dialog](screenshots/Export%20classification%20dataset.png) | ![Classification export summary](screenshots/Report%20of%20Export%20classification%20dataset.png) |
| *Exporting — pick a format, split ratio, and output folder* | *Confirmation once the export finishes* |
| ![Exported classification dataset on disk](screenshots/Exported%20classification%20dataset.png) | |
| *Result: a train/val/test ImageFolder layout on disk* | |

### Detection

| | |
|---|---|
| ![Detection workspace after auto-labeling](screenshots/Detection%20dataset%20after%20Auto-Labeling.png) | ![Auto-Label Settings for Detection](screenshots/Auto-Label%20Settings%20-%20Detection.png) |
| *Bounding boxes drawn automatically across a batch of images* | *Model, device (CPU/GPU), and confidence threshold* |
| ![Auto-labeling progress for Detection](screenshots/Auto-Label%20Report%20-%20Detection.png) | ![Export Detection dataset dialog](screenshots/Export%20detection%20dataset.png) |
| *Live progress while auto-labeling runs* | *Exporting to YOLO, Pascal VOC, or CreateML* |
| ![Detection export summary](screenshots/Report%20of%20Export%20detection%20dataset.png) | ![Exported detection dataset on disk](screenshots/Exported%20detection%20dataset.png) |
| *Confirmation once the export finishes* | *Result: images + label files, split by train/val/test* |

### Segmentation

| | |
|---|---|
| ![Segmentation workspace after auto-labeling](screenshots/Segmentation%20dataset%20after%20Auto-Labeling.png) | ![Auto-Label Settings for Segmentation](screenshots/Auto-Label%20Settings%20-%20Segmentation.png) |
| *Polygon masks traced automatically across a batch of images* | *Model, device, and confidence threshold* |
| ![Auto-labeling progress for Segmentation](screenshots/Auto-Label%20Report%20-%20Segmentation.png) | ![Export Segmentation dataset dialog](screenshots/Export%20segmentation%20dataset.png) |
| *Live progress while auto-labeling runs* | *Exporting to YOLO-seg or COCO* |
| ![Segmentation export summary](screenshots/Report%20of%20Export%20segmentation%20dataset.png) | ![Exported segmentation dataset labels folder](screenshots/Exported%20segmentation%20dataset%20labels%20path.png) |
| *Confirmation once the export finishes* | *Result: a labels/ folder of polygon annotations, split by train/val/test* |

## Installation

**Requirements:** Python 3.12 or newer.

```bash
# 1. Clone or download this repository, then from its root:
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. Install LabelMaker and its base dependencies
pip install -e .

# 3. Launch it
labelmaker
```

The base install covers Classification/Detection/Segmentation labeling and ONNX-based auto-labeling out of the box. A few pieces are optional because they're large packages you may not need:

| Extra | Installs | Needed for |
|---|---|---|
| `pip install -e ".[torch]"` | PyTorch | Loading `.pt`/`.pth` TorchScript models |
| `pip install -e ".[ultralytics]"` | PyTorch + Ultralytics | Loading raw Ultralytics YOLO checkpoints (`.pt`) |
| `pip install -e ".[torchvision]"` | PyTorch + torchvision | Reconstructing a Classification `state_dict` checkpoint |
| `pip install -e ".[gpu]"` | `onnxruntime-gpu` | GPU-accelerated auto-labeling for **ONNX** models on an NVIDIA GPU |

> `onnxruntime` and `onnxruntime-gpu` conflict if both are installed — run `pip uninstall onnxruntime` first if you're adding the `gpu` extra to an existing install.

For GPU-accelerated PyTorch/Ultralytics auto-labeling, install a CUDA build of PyTorch directly instead (pip's default index serves a CPU-only build):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

(swap `cu126` for whichever CUDA tag matches your installed NVIDIA driver — see [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)). Either way, GPU is only ever used when you explicitly select it in Auto-Label Settings — never automatically.

## Quick Start

1. Launch `labelmaker` and pick a task on the home screen — **Classification**, **Detection**, or **Segmentation** — or open a project you already started.
2. Name your project. This becomes a folder (under `projects/`) that holds its images, classes, and labels.
3. **Import Folder** brings in every image in a folder (and its subfolders); **Import Image** adds specific files; **Import Classes** loads a plain-text file of class names, one per line.
4. Click **+ Add Class** to define what you're labeling with — give each one a name, a color, and an optional single-key shortcut.
5. Label your images — see [Manual Labeling](#manual-labeling) below — or click **Auto-Label Settings…** to label with a model instead, see [Auto-Labeling](#auto-labeling).
6. Click **Export Dataset** to save a training-ready copy — see [Exporting a Dataset](#exporting-a-dataset).

Everything autosaves as you go; there's no separate Save step.

## Manual Labeling

<details>
<summary><b>Classification</b> — one class per image</summary>

1. Select one or more images in the grid (click, or Ctrl/Shift-click for several).
2. Click a class button to make it the *active* class (only one is active at a time).
3. Click **Assign (Enter)**, or just press Enter. The selected images are labeled immediately.
4. The **Mode** toggle controls what happens to the file: **Copy** (default) places a copy in a folder named after the class and leaves the original alone; **Move** relocates the original there instead.
5. Use the **Filter** dropdown to switch between *Unlabeled* (what's left to do), *All*, and *By Class* (review one class at a time).

</details>

<details>
<summary><b>Detection</b> — bounding boxes</summary>

1. Click a class to make it active — this decides the class of the *next* box you draw, not any existing ones.
2. Click and drag on the image to draw a box around an object.
3. Click an existing box to select it: drag its body to move it, drag a corner/edge to resize, press **Delete** to remove it.
4. The **Objects in this image** panel lists every box on the current image — use it to select, reassign, or delete without touching the canvas.

</details>

<details>
<summary><b>Segmentation</b> — polygon outlines</summary>

1. Click a class to make it active, then click once per corner/curve as you work your way around the object's edge.
2. Close the shape by clicking back on the first point (it turns green once you're close), pressing **Enter**, or double-clicking. You need at least 3 points.
3. Click a shape to select it: drag its body to move it, drag a point to reshape it, double-click a point to delete just that point.
4. Just like Detection, the **Objects in this image** panel lets you select, reassign, or delete any shape.

</details>

## Auto-Labeling

Every task has a matching auto-labeler: point it at a model you've already trained (or a pretrained one), and it labels your images for you.

1. Click **Auto-Label Settings…**, browse to your model file, and choose a **Device** (CPU or the GPU dropdown, if one's detected and usable).
2. Loading the model automatically lists its declared classes; **Save Settings** adds any new ones to your project's class list.
3. Set a **confidence threshold** and whether to run on *unlabeled images only* (the safe default) or *all images*.
4. Click **Start Auto-Labeling** and watch the live progress — you can cancel at any point without losing what's already been labeled.

**Supported model formats**, auto-detected from the file:

| Format | Extension | Notes |
|---|---|---|
| ONNX | `.onnx` | Recommended — self-contained, works out of the box, and is what GPU support (`onnxruntime-gpu`) targets. |
| Darknet | `.weights` + `.cfg` | Classic YOLOv3/v4 weights, via OpenCV's DNN module. |
| Caffe | `.caffemodel` + `.prototxt` | Via OpenCV's DNN module. |
| TensorFlow | `.pb` (frozen graph) | Via OpenCV's DNN module. |
| PyTorch TorchScript | `.pt` / `.pth` | Needs the `torch` extra. |
| Ultralytics YOLO checkpoint | `.pt` | Needs the `ultralytics` extra. `detect`/`classify`/`segment` checkpoints each route to the matching workspace automatically. |
| PyTorch `state_dict` | `.pt` / `.pth` | Classification only — reconstructs a known torchvision architecture (ResNet, VGG, MobileNet, ...) from a checkpoint that names its own architecture. Needs the `torchvision` extra. |

The Segmentation auto-labeler accepts both semantic segmentation models (a per-pixel class map) *and* Ultralytics YOLO-seg checkpoints (per-instance masks) — either way, you get polygon shapes on your images.

## Exporting a Dataset

Click **Export Dataset**, choose a format, set your train/val/test split percentages, and pick an output folder. Exporting only writes new files there — your project itself is never modified.

| Task | Format | Layout |
|---|---|---|
| **Classification** | ImageFolder | `{train,val,test}/{class_name}/*.jpg` — what `torchvision.datasets.ImageFolder`, Keras' `flow_from_directory`, and most classifiers expect directly. |
| | CSV manifest | `images/{train,val,test}/*.jpg` + a `{split}.csv` (`filename,label`) per split. |
| **Detection** | YOLO | `images/` + `labels/` (`.txt`, normalized boxes) split by train/val/test, plus `classes.yaml`. |
| | Pascal VOC | `JPEGImages/` + `Annotations/` (one `.xml` per image) + `ImageSets/Main/{split}.txt`. |
| | CreateML | One folder per split, each with its images plus a single `annotations.json`. |
| **Segmentation** | YOLO-seg | `images/` + `labels/` (`.txt`, one normalized polygon per line) split by train/val/test, plus `classes.yaml`. |
| | COCO | `{train,val,test}/` image folders + `annotations/instances_{split}.json` (polygon `segmentation`, `bbox`, `area`). |

Only labeled images are exported — anything without a class/box/polygon yet is skipped and reported in the export summary.

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `F` | Fit the current image to the view |
| Mouse wheel | Zoom in/out over an image |
| `A` / `D` | Previous / next image in the grid |
| `1`–`9` | Activate a class with that shortcut key |
| `Delete` / `Backspace` | Remove selected image(s) from the project, or the selected box/shape |
| `Escape` | Cancel a segmentation shape you're still drawing |
| `Ctrl+N` | Go home (leave the current project) |
| `Ctrl+O` | Open an existing project |

Removing an image from a project never deletes the file on disk — it only stops tracking it.

## Project Structure

```
LabelMaker/
├── app/                  # Application source
│   ├── core/             # Main window, toolbar, app shell
│   ├── features/         # classification/, detection/, segmentation/, project/
│   └── shared/           # Reusable UI widgets, theming, auto-label engine
├── projects/             # Your projects live here (created on first run)
├── settings/             # App-level settings (created on first run)
└── pyproject.toml
```

Each project is a self-contained folder under `projects/` holding a `project.json` (classes, image list, labels) alongside your images (for Classification's Copy/Move modes).

## License

Released under the [MIT License](LICENSE).
