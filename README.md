# When Regularization Fails: A Study in Image Stitching

This project investigates the unintended effects of regularization methods (L2, Dropout, Early Stopping) on CNN-based local feature descriptor models in keypoint-based homography estimation and image stitching. The goal is to identify conditions under which regularization degrades descriptor performance, resulting in poor geometric alignment or unstable homographies.

---

## 📂 Project Structure

```
project/
├── build_scaffold.py          # Creates all necessary folders
├── environment.yaml           # Conda environment definition
├── README.md                  # Project overview and usage
├── report/                    # Final report in notebook and/or PDF format
├── data/                      # COCO source + generated image pairs
│   ├── raw/                   # COCO 2014 train/val images
│   ├── processed/             # Resized / grayscale / transformed pairs
│   └── pairs.json             # Manifest mapping original to warped views
├── notebooks/                 # Development notebooks
├── scripts/                   # Optional one-off helpers (augmentation, logs)
├── src/                       # Core modular Python code
│   ├── data/                  # Loading and preprocessing logic
│   ├── models/                # CNN descriptor model definitions
│   ├── homography/            # Matching, RANSAC, SVD analysis
│   └── evaluation/            # Metrics (RMSE, SSIM, condition number)
└── outputs/                   # Logs, trained models, mosaic results
```

---

## 📊 Final Report Table of Contents

1. Title & Executive Summary  
2. Introduction & Problem Statement  
3. Background & Theory  
4. Dataset & Preprocessing  
5. Synthetic View Generation  
6. CNN Descriptor Architectures  
7. Training Setup  
8. Feature Matching + Homography Estimation  
9. SVD & Conditioning Analysis  
10. Mosaic Reconstruction + SSIM  
11. Results  
12. Discussion  
13. Conclusion  
14. References  
15. Appendix  

---

## 📖 Usage

### Setup

```bash
conda env create -f environment.yaml
conda activate regenv
python build_scaffold.py
```

### Run Development Workflow

- Follow notebooks in `notebooks/` to run each part of the pipeline.
- Final results and stitched mosaics are saved in `outputs/`.

---

## 🎓 Author

Brandon Henley  
CSCI E-25: Computer Vision, Harvard Extension School