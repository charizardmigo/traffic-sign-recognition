
# Traffic Sign Recognition System

An end-to-end computer vision pipeline for real-time traffic sign detection and classification in dash-cam video footage. This system combines a lightweight NanoDet object detector with a custom-trained classifier to identify and annotate road signs (speed limits, stop signs, parking signs, turn directions, etc.) on CPU-only hardware, making it suitable for edge deployment in ADAS and autonomous driving applications.

---

## 🎯 Project Overview

This project was developed as a final semester capstone to demonstrate practical skills in:

- Building production-ready computer vision pipelines from scratch
- Training custom object detection models on domain-specific datasets
- Optimizing models for real-time inference on resource-constrained hardware
- Creating reproducible ML workflows with clear documentation

**Key achievement:** Real-time traffic sign detection and classification on CPU-only systems, demonstrating feasibility for deployment in cost-sensitive automotive applications.

---

## 🚀 Features
- **Custom Object Detection:** NanoDet-Plus-m-1.5x trained on a COCO-format traffic sign dataset
- **10-Class Recognition:** Detects tunnel, speed limits (50/100), intersections, parking, directional signs, construction zones, stop signs, and traffic lights
- **Real-Time Inference:** Optimized for CPU-only execution with minimal latency
- **Video Processing Pipeline:** Frame-by-frame annotation with inference time tracking
- **Flexible Training Configuration:** YAML-based hyperparameter configuration (my_dataset.yml)
- **Model Conversion Utilities:** Checkpoint conversion from PyTorch Lightning .ckpt to standard PyTorch .pt
- **Reproducible Dataset Pipeline:** YOLO → train/val/test split → COCO conversion for NanoDet

---

## 🛠️ Tech Stack

### Languages & Core

- Python 3.12.3

### Deep Learning and CV Frameworks

- PyTorch (NanoDet detector training & inference)
- PyTorch Lightning (training orchestration, checkpointing)
- TensorFlow 2 / Keras (traffic sign classifier experiments)
- NanoDet (lightweight one-stage object detector)
- OpenCV (video processing, frame handling, basic image ops)

### Data Handling & Utilities

- NumPy
- Pandas
- pycocotools / COCO API (annotation handling)
- Custom YOLO → COCO conversion and dataset preparation scripts
- Training, Evaluation and Experimentation
- scikit-learn (confusion matrices and basic metrics for the classifier)
- TensorBoard (training monitoring)

---

## 📁 Project Structure

**Note:** This is a representative structure. Some utility files and framework internals are omitted for brevity.

```text
traffic-sign-recognition/
├── code/                              # Core project scripts
│   ├── dataset_preparation.py         # Split YOLO dataset + convert to COCO
│   ├── convert_ckpt_to_pt.py          # Convert .ckpt → .pt for inference
│   ├── NanoDETVideoPlayer.py          # Tkinter-based video inference GUI
│   ├── yolo_to_coco.py                # Standalone YOLO → COCO utility (legacy)
│   └── diagnose_ckpt.py               # (Dev utility) Inspect/checkpoint diagnostics
├── data/
│   ├── datasets/
│   │   ├── dataset_original/          # Raw YOLO-format dataset (images + labels)
│   │   ├── dataset_test/              # (Generated) YOLO train/val/test splits
│   │   └── dataset_mod_coco/          # (Generated) COCO-format dataset for NanoDet
│   └── video/
│       └── AutoraceTrackLab61All.mp4  # Example test video
├── nanodet/                           # NanoDet framework (submodule / vendor code)
│   ├── config/
│   │   └── my_dataset.yml             # Training config for this dataset
│   ├── tools/
│   │   └── train.py                   # Standard NanoDet training entrypoint
│   └── ...                            # Backbones, heads, utils, etc.
├── workspace/
│   └── nanodet-trained-model/         # Generated after training
│       ├── model_last.ckpt            # PyTorch Lightning checkpoint (training output)
│       └── nanodet_model.pt           # Converted model weights for inference
├── requirements.txt
└── README.md
```
---

## ⚙️ Installation

### Prerequisites

- **OS:** Ubuntu 24.04 LTS (tested), macOS, or Windows 10/11
- **Python:** 3.12.3 (recommended) or 3.9+
- **Hardware:** CPU-only training and inference supported. GPU optional for faster training

### 1. Clone the Repository

```bash
git clone https://github.com/charizardmigo/traffic-sign-recognition.git
cd traffic-sign-recognition
```

### 2. Create a Virtual Environment

**Linux / macOS:**

```bash
python3 -m venv .nanodet_env
source .nanodet_env/bin/activate
```

**Windows:**

```cmd
python -m venv .nanodet_env
.nanodet_env\Scripts\activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install NanoDet as Editable Module

```bash
cd nanodet
pip install -e .
cd ..
```

---

## 📊 Dataset Preparation

This project uses a two-stage dataset preparation process. First, splitting the raw YOLO-formatted data into `train/validation/test` sets, and then converting these splits into COCO format, which is required by NanoDet.

The `code/dataset_preparation.py` script handles both steps.

### Step 1: Place Your Raw Data

Place your raw YOLO-formatted dataset (images + .txt label files) into:

```text
data/datasets/dataset_original/
```

This directory should ideally contain `images/` and `labels/` subdirectories, or a structure that `dataset_preparation.py` can automatically detect (it supports common Roboflow-style exports).

**Important notes:**

- `dataset_preparation.py` reads from `data/datasets/dataset_original/` but does not modify its contents.
- To use your own dataset, simply replace the contents of `data/datasets/dataset_original/` with your images and labels, then run the script as shown below.

### Step 2: Split and Convert to COCO Format

From the project root, run:

```bash
python3 code/dataset_preparation.py \
    data/datasets/dataset_original/ \
    data/datasets/dataset_test/ \
    --create_coco
```

This command will split the raw YOLO dataset (`dataset_original/`) into train, val, and test subsets (default ratios: 70% / 20% / 10%) and save them in YOLO format to:

```text
data/datasets/dataset_test/
  ├── train/
  │   ├── images/
  │   └── labels/
  ├── val/
  │   ├── images/
  │   └── labels/
  └── test/
      ├── images/
      └── labels/
```

Then it converts these splits to COCO format and save the final dataset to:

```text
data/datasets/dataset_mod_coco/
  ├── images/
  │   ├── train/
  │   └── val/
  ├── train.json
  └── val.json
```

### Using Your Own Dataset

You have two options:

**A: Replace the original dataset:**

- Delete or move the existing contents of `data/datasets/dataset_original/`.
- Copy your own images and YOLO labels into `data/datasets/dataset_original/`.
- Run `dataset_preparation.py` with `--create_coco` as in Step 2 above.

**B: Extend the original dataset:**

- Leave the existing files in `dataset_original/`.
- Add your images and labels alongside them.
- Run `dataset_preparation.py` again to generate a combined dataset.

---

## 🏋️ Training the Detector

Training is handled by NanoDet’s standard training script using your custom config.

### 1. Inspect / Edit Training Configuration

The NanoDet configuration is stored in:

```text
nanodet/config/my_dataset.yml
```

Key aspects:

```yaml
Backbone: ShuffleNetV2 (1.5x)
Neck: GhostPAN
Head: NanoDetPlusHead
Input Size: 320 x 320 pixels
Number of Classes: 10, matching:
class_names: ['tunnel', 'Speedlimit_50', 'Speedlimit_100',
              'intersection', 'parking', 'right', 'left',
              'construction', 'stop', 'traffic_light']

device:
  gpu_ids: -1        # -1 for CPU, 0 for single GPU, [0,1] for multi-GPU
  workers_per_gpu: 2
  batchsize_per_gpu: 16
  precision: 32      # set 16 for AMP (GPU only)

schedule:
  total_epochs: 10   # increase for better performance
  optimizer:
    name: AdamW
    lr: 0.001
```

You may want to adjust:

- Data paths (if you change dataset folders).
- Augmentation settings (scale, flip, brightness, contrast, etc.).
- Training schedule (epochs, learning rate, optimizer).
- Device settings (CPU vs GPU, batch size, precision).

### 2. Start Training

From the project root, run:

```bash
python3 nanodet/tools/train.py nanodet/config/my_dataset.yml
```

Checkpoints and logs will be saved in:

```text
workspace/nanodet-trained-model/
```

---

## 🔄 Model Conversion (.ckpt → .pt)

NanoDet with PyTorch Lightning saves checkpoints as `.ckpt` files. The video player and many inference scripts expect a plain PyTorch `state_dict` in `.pt` format.

Run the following command in project root:

```bash
python3 code/convert_ckpt_to_pt.py \
    workspace/nanodet-trained-model/model_last.ckpt \
    workspace/nanodet-trained-model/nanodet_model.pt
```

This command does the following:

- Loads the Lightning checkpoint.
- Extracts the `state_dict` containing only the model weights.
- Saves it as `nanodet_model.pt`, which can be loaded by the inference GUI.

---

## 🎥 Inference & Testing (Video Player)

The project includes a Tkinter-based GUI (NanoDETVideoPlayer) for running the detector on videos or a camera feed.

### 1. Launch the Player

```bash
python3 code/NanoDETVideoPlayer.py
```

### 2. Load the Trained Model

Click “Change YOLO Model”, and navigate to the model you just converted in:

```text
workspace/nanodet-trained-model/nanodet_model.pt
```

### 3. Load a Video or Camera

Ensure “Video File” is selected, then click “Open Video” and choose a file. You can use the video sample already provided in:

```text
data/video/AutoraceTrackLab61All.mp4
```

Alternatively, you can select “Camera” as the source for testing. The player will attempt to open the default camera index.

### 4. Enable Processing and Saving (Optional)

- Check “Enable Frame Processing” to run NanoDet on each frame. 
- The terminal will print per-frame inference time in milliseconds.
- You can also check “Enable Output File Generation” to save annotated video.
- You will be prompted to choose an output file path (e.g. output.avi).

### 5. Controls

- **Play / Pause:** Start or pause playback.
- **Stop:** Reset to the beginning.
- **<< / >>:** Step backwards/forwards.
- **Slider**: Seek to a specific frame (video file mode only).
- **Snapshot:** Save the current frame as a JPEG in `snapshots/`.

---

## 🔧 Troubleshooting

### 1. ModuleNotFoundError: No module named 'nanodet'

Ensure you installed NanoDet in editable mode:

```bash
cd nanodet
pip install -e .
cd ..
```

### 2. No images / labels found in dataset_preparation.py

Check that `data/datasets/dataset_original/` contains `images/` and `labels/` subfolders, or a similar structure (e.g. Roboflow export) that the script can detect.

### 3. Video Player cannot load model or crashes on startup

Confirm that `nanodet_model.pt` exists and was created using `convert_ckpt_to_pt.py`.
Use “Change YOLO Model” on Video Player to explicitly select the `.pt` file.

### 4. Very low FPS during inference

- Run on smaller videos or downscale frames.
- Consider reducing input resolution in the config.
- For real deployment, consider GPU or further model optimization (ONNX/TensorRT, quantization).

---

## 🚀 Possible Future Work

- Export model to ONNX / TensorRT for faster inference.
- Integrate tracking to maintain object identity across frames.
- Support live ROS topics or other robotics middleware.
- Extend the dataset with more sign types and more challenging scenarios.
- Quantization and pruning for deployment on embedded devices (Jetson Nano, Raspberry Pi).

---

## 📄 License

This project builds on the NanoDet framework, which is licensed under the **Apache License 2.0.**  
Please consult the NanoDet repository and respect all associated licenses for any derivative work.  
All additional code, configuration, and training scripts in this repository are provided under the **MIT License.**  
See the `LICENSE` file for details.

---

## 👤 Author

Priestley Fomeche  
GitHub: [@charizardmigo](https://github.com/charizardmigo)  
LinkedIn: [Priestley Fomeche](https://linkedin.com/in/priestley-fomeche)  
Email: fomechepriestly7@gmail.com

---

## 🙏 Acknowledgments

[NanoDet](https://github.com/RangiLyu/nanodet) by RangiLyu for the lightweight detection framework.  
[PyTorch Lightning](https://lightning.ai/) for simplifying the training loop.  
[COCO Dataset Format](https://cocodataset.org/) for providing a standard annotation scheme.

---

## 📚 References

[NanoDet GitHub Repository](https://github.com/RangiLyu/nanodet)  
[PyTorch Documentation](https://pytorch.org/docs/stable/index.html)  
[PyTorch Lightning Docs](https://lightning.ai/docs/pytorch/stable/)  
[COCO Format Explained](https://cocodataset.org/#format-data)

---

⭐ If you find this project useful, please consider giving it a star!