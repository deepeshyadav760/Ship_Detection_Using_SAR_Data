# Ship Detection in Sentinel-1 SAR Imagery

This repository implements an end-to-end pipeline for ship detection using Sentinel-1 Synthetic Aperture Radar (SAR) imagery. The workflow includes SAR data acquisition via Google Earth Engine, dataset preparation, YOLOv8 model training, and inference on real-world maritime scenes.

---

## Repository Structure

```
GALAXEYE_SHIP_DETECTION/
├── data/
│   ├── gee_export/
│   │   └── sentinel1_vv_panama.tif     # Sentinel-1 SAR export from GEE (not committed)
│   ├── raw/                             # Original SSDD dataset
│   ├── yolo/                            # SSDD converted to YOLO format
│   └── ship.yaml                        # YOLO dataset configuration
├── GalaxEye_venv/                       # Local virtual environment (ignored)
├── report/                              # Final report (PDF)
├── results/
│   ├── label_check/                     # Label visualization sanity checks
│   └── sentinel1_inference/             # Final inference outputs
├── runs_detect/                         # YOLO detection outputs (training/inference)
│   ├── predict/
│   ├── runs/
│   ├── training/
│   └── val/
├── scripts/
│   ├── convert_ssdd_to_yolo.py         # SSDD → YOLO conversion
│   ├── sentinel1_download_gee.ipynb    # Sentinel-1 download using GEE
│   ├── sentinel1_yolo_inference.ipynb  # Inference on Sentinel-1 SAR
│   ├── train_yolov8_sssd.ipynb         # YOLOv8 training notebook
│   └── visualize_yolo_labels.py        # Dataset label verification
├── .gitignore
├── requirements.txt
├── README.md
└── yolov8s.pt                           # Pretrained YOLOv8 weights
```

---

## Dataset

### SSDD (SAR Ship Detection Dataset)
- Used for model training and validation
- Contains annotated SAR ship image patches
- Converted into YOLO format for object detection training

### Sentinel-1 GRD
- Used for real-world inference
- VV polarization
- Interferometric Wide (IW) mode
- Data accessed programmatically via Google Earth Engine

---

## Methodology

### 1. Sentinel-1 Data Acquisition
- Sentinel-1 GRD images downloaded using Google Earth Engine Python API
- Filters applied:
  - IW acquisition mode
  - VV polarization
  - Descending orbit pass
- Temporal averaging applied to reduce speckle noise
- Exported as GeoTIFF for local inference

### 2. SAR Preprocessing
- Conversion to decibel (dB) scale
- Clipping to the range [-25 dB, 0 dB]
- Handling of NaN and infinite values
- Normalization and conversion to uint8
- SAR image replicated to 3 channels for YOLO compatibility

### 3. Model Training
- YOLOv8 used as the object detection framework
- SSDD dataset converted to YOLO format
- Model trained for 100 epochs
- High detection performance achieved on validation data

### 4. Inference on Sentinel-1 SAR
- Sliding-window inference with adaptive tiling
- Image padding to ensure sufficient spatial context
- Global Non-Maximum Suppression (NMS) applied
- Inference performed on a high-density maritime region near the Panama Canal

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU for the training

### Setup

1. Clone the repository:
```bash
git clone https://github.com/deepeshyadav760/GALAXEYE_SHIP_DETECTION.git
cd GALAXEYE_SHIP_DETECTION
```

2. Create and activate virtual environment:
```bash
python -m venv GalaxEye_venv
source GalaxEye_venv/bin/activate  # On Windows: GalaxEye_venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Google Earth Engine (for Sentinel-1 download):
```bash
earthengine authenticate
```

---

## Usage

### 1. Download Sentinel-1 Data
```bash
jupyter notebook scripts/sentinel1_download_gee.ipynb
```
Follow the notebook to download Sentinel-1 imagery for your region of interest.

### 2. Convert SSDD Dataset to YOLO Format
```bash
python scripts/convert_ssdd_to_yolo.py
```

### 3. Visualize Dataset Labels
```bash
python scripts/visualize_yolo_labels.py
```

### 4. Train YOLOv8 Model
```bash
jupyter notebook scripts/train_yolov8_sssd.ipynb
```

### 5. Run Inference on Sentinel-1 SAR
```bash
jupyter notebook scripts/sentinel1_yolo_inference.ipynb
```

---

## Results

Final inference results are stored in:
```
results/sentinel1_inference/
```

The model successfully detects ships in Sentinel-1 SAR imagery with tight bounding boxes and minimal false positives.

### Performance Metrics
- High precision on maritime vessel detection
- Robust performance across varying sea states
- Effective speckle noise handling through preprocessing

---

## Notes

- Large Sentinel-1 `.tif` files are not committed to the repository
- All results are fully reproducible using the provided notebooks
- The pipeline is designed to be modular and extensible
- Pretrained YOLOv8 weights (`yolov8s.pt`) are included for quick start

---

## Future Work

- Rotated bounding boxes for elongated vessels
- Multi-temporal ship density analysis
- AIS-based validation of detected vessels
- Extension to multi-polarization SAR data (VH polarization)
- Real-time detection pipeline for maritime surveillance
- Integration with ship classification models

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- **SSDD Dataset**: SAR Ship Detection Dataset contributors
- **Sentinel-1**: European Space Agency (ESA) Copernicus Programme
- **Google Earth Engine**: For providing accessible SAR data infrastructure
- **Ultralytics YOLOv8**: For the state-of-the-art object detection framework

---

## Author

**Deepesh Yadav**

---

## References

- Sentinel-1 SAR User Guide: https://sentinel.esa.int/web/sentinel/user-guides/sentinel-1-sar
- YOLOv8 Documentation: https://docs.ultralytics.com/
- Google Earth Engine: https://earthengine.google.com/