# IndiVNet: A Region Adaptive Semantic Image Segmentation for Autonomous Driving in Unstructured Environments

A lightweight, region-adaptive semantic image segmentation network tailored for complex, unstructured driving scenes using a progressive dilation encoder (6→16).

---

## 1. Paper, DOI & Citation

- **Journal**: *Scientific Reports* 16, 2468 (2026)
- **DOI**: [10.1038/s41598-025-32305-2](https://doi.org/10.1038/s41598-025-32305-2)
- **Paper Link**: [https://doi.org/10.1038/s41598-025-32305-2](https://doi.org/10.1038/s41598-025-32305-2)
- **Authors**: Pritam Chakraborty, Anjan Bandyopadhyay, Siddhartha Bhattacharyya, and Jan Platos

```bibtex
@article{chakraborty2026indivnet,
  title     = {IndiVNet: A region adaptive semantic image segmentation for autonomous driving in unstructured environments},
  author    = {Chakraborty, Pritam and Bandyopadhyay, Anjan and Bhattacharyya, Siddhartha and Platos, Jan},
  journal   = {Scientific Reports},
  volume    = {16},
  number    = {1},
  pages     = {2468},
  year      = {2026},
  publisher = {Nature Publishing Group UK London},
  doi       = {10.1038/s41598-025-32305-2},
  url       = {https://doi.org/10.1038/s41598-025-32305-2}
}
```

---

## 2. Results

All numbers below match the published paper exactly.

### Benchmark Comparison on India Driving Dataset (IDD)
*Evaluated on finely annotated IDD test set with 95% bootstrap confidence intervals:*

| Model | Input Resolution | mIoU (%) | Dice Score (%) | Parameters (M) | FLOPs (G) | FPS (256×256) | Memory (GB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **IndiVNet (Proposed)** | **256 × 256** | **69.98 ± 0.2** | **82.4 ± 0.3** | **43.7** | **28.6** | **15** | **1.2** |
| BSNet | 720 × 960 | 62.1 ± 0.4 | 76.7 ± 0.4 | 41.1 | — | 11 | — |
| LETNet | 360 × 480 | 61.8 ± 0.4 | 76.4 ± 0.4 | — | — | 14 | — |
| FBSNet | 360 × 480 | 59.7 ± 0.4 | 74.8 ± 0.5 | — | — | 17 | — |
| SGCPNet | 360 × 480 | 58.3 ± 0.5 | 73.7 ± 0.5 | — | — | 18 | — |
| LEANet | 360 × 480 | 57.4 ± 0.5 | 72.9 ± 0.6 | — | — | 16 | — |
| ICNet | 256 × 256 | 56.20 | — | 26.5 | 28.9 | 22 | 1.1 |
| U-Net | 256 × 256 | 55.30 | — | 31.4 | 39.8 | 12 | 1.5 |
| LAANet | 360 × 480 | 51.6 ± 0.6 | 68.1 ± 0.7 | — | — | 18 | — |
| BiSeNetV2 | 256 × 256 | 65.30 | — | 49.0 | 35.7 | 18 | 1.4 |
| HRNetV2 | 256 × 256 | 68.20 | — | 65.8 | 48.3 | 6 | 2.4 |
| DeepLabV3+ | 256 × 256 | 43.80 | — | 54.1 | 45.2 | 9 | 1.8 |

### Cross-Domain Generalization on CamVid
| Model | Input Resolution | mIoU (%) | Dice Score (%) |
| :--- | :---: | :---: | :---: |
| **IndiVNet (Proposed)** | **256 × 256** | **73.2 ± 0.2** | **84.5 ± 0.3** |
| BSNet | 720 × 960 | 72.5 ± 0.3 | 84.1 ± 0.3 |
| LETNet | 360 × 480 | 70.5 ± 0.3 | 83.1 ± 0.3 |
| SGCPNet | 360 × 480 | 69.0 ± 0.4 | 81.7 ± 0.4 |
| FBSNet | 360 × 480 | 68.9 ± 0.4 | 81.6 ± 0.4 |
| LEANet | 360 × 480 | 67.5 ± 0.4 | 80.6 ± 0.5 |
| LAANet | 360 × 480 | 67.5 ± 0.4 | 80.6 ± 0.5 |

### Per-Class Evaluation on IDD Test Set
| Class | TP | FP | TN | FN | F1-Score | Accuracy | Class IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Drivable** | 25,008 | 655 | 102,077 | 446 | 0.9850 | 0.9914 | **0.958** |
| **Sky** | 16,878 | 213 | 6,341 | 1,305 | 0.9015 | 0.9386 | **0.917** |
| **Vehicle** | 2,134 | 634 | 3,091 | 446 | 0.8192 | 0.8287 | **0.664** |
| **Far Object** | 10,420 | 1,949 | 59,983 | 3,498 | 0.9907 | 0.9282 | **0.657** |
| **Roadside Object** | 2,052 | 2,005 | 3,513 | 2,911 | 0.5255 | 0.5310 | **0.294** |
| **Overall Mean** | — | — | — | — | — | — | **0.6998 (mIoU)** |

---

## 3. Setup

### Python and Framework Versions
- **Python**: 3.8+ (tested on Python 3.8 - 3.10)
- **TensorFlow**: 2.12.0

### Installation
```bash
git clone https://github.com/Pritamresearch/IndiVnet.git
cd IndiVnet

# Create and activate virtual environment
python -m venv venv
# Windows: venv\Scripts\activate    Linux/macOS: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### `requirements.txt`
```text
imutils==0.5.4
tensorflow==2.12.0
opencv-python==4.7.0.72
matplotlib==3.7.1
numpy==1.24.3
tqdm==4.65.0
```

---

## 4. Dataset Instructions

The model is trained and benchmarked on the **India Driving Dataset (IDD)** / **IDD-Lite**:
- **Dataset Portal**: [http://idd.insaan.iiit.ac.in/](http://idd.insaan.iiit.ac.in/)
- **Annotations**: 10,003 finely annotated images across 182 driving sequences (6,993 train, 981 val, 2,029 test).

### Expected Directory Layout
```text
idd20k_lite/
├── leftImg8bit/
│   ├── train/
│   │   ├── 001/
│   │   │   ├── 000123_image.jpg
│   │   │   └── ...
│   └── val/
└── gtFine/
    ├── train/
    │   ├── 001/
    │   │   ├── 000123_label.png
    │   │   └── ...
    └── val/
```

In `Data Loading and Preprocessing/data.py`, update `dataset_path` to your local dataset directory:
```python
dataset_path = 'path/to/idd20k_lite/'
```

*Note on Class Remapping*: In `data.py`, pixels assigned `255` ("not labeled") are dynamically remapped to label `7` to maintain consistent 7-class indexing (0–6 foreground, 7 background).

---

## 5. Training and Evaluation Commands

All commands below exist directly in the repository scripts.

### Training
To train the model using custom `tf.GradientTape` training loop with Adam optimizer and checkpointing:
```bash
python Train/train.py
```
Or execute through the unified pipeline:
```bash
python "IndiVnet Main/main.py"
```

### Evaluation
To evaluate trained weights and print class-wise IoU and overall mIoU:
```bash
python Test/test.py
```

### Python API Inference Snippet
```python
import tensorflow as tf
import numpy as np
import cv2
from Model.model import IndiVnet

# 1. Instantiate model
model = IndiVnet(n_classes=8, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=3)

# 2. Load weights
model.load_weights('IndiVnet_weight-epoch-43.hdf5')

# 3. Preprocess and predict
img = cv2.imread("sample.jpg")
img_resized = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (256, 256)) / 255.0
pred = model.predict(np.expand_dims(img_resized, axis=0))
mask = np.argmax(pred[0], axis=-1)
```

---

## 6. Pretrained Weights

- **Status**: Weights: to be released.
- The evaluation script `Test/test.py` expects `IndiVnet_weight-epoch-43.hdf5`. Download links will be posted upon external repository hosting.

---

## 7. Hardware Tested

Taken directly from Section 4 ("Experimental setup") of the published paper:
- **Environment**: Google Colaboratory infrastructure
- **Compute**: 15.0 GB GPU Memory (NVIDIA GPU), 12.7 GB System RAM, 78.2 GB available disk storage
- **Inference Speed**: 15 FPS at 256×256 input resolution
- **Model Size**: 43.7M parameters, 28.6 GFLOPs, 1.2 GB GPU memory footprint during inference

---

## 8. Licence

Licence: To be confirmed by author. (Please contact Pritam Chakraborty regarding licensing terms).
