# When Regularization Fails: A Study in Image Stitching

This project investigates the unintended effects of regularization methods (L2, Dropout, Early Stopping) on CNN-based local feature descriptor models in keypoint-based homography estimation and image stitching. The goal is to identify conditions under which regularization degrades descriptor performance, resulting in poor geometric alignment or unstable homographies.

---

## Project Structure

```
project/
├── build_scaffold.py          # Creates all necessary folders
├── environment.yaml           # Conda environment definition
├── README.md                  # Project overview and usage
├── report/                    # Final report in notebook and/or PDF format
├── data/                      # COCO source + generated image pairs
│   ├── raw/                   # COCO 2014 train/val images
│   ├── processed/             # Resized / grayscale / transformed pairs
│   ├── selected_train/        # Random sample used for training
│   ├── selected_eval/         # Random sample used for evaluation
│   ├── pairs_train.json       # Manifest mapping original → warped (train)
│   └── pairs_eval.json        # Manifest mapping original → warped (eval)
├── notebooks/                 # Jupyter notebooks for full experiments
├── scripts/                   # Scripts for data generation or helpers
├── src/                       # Modular Python code
│   ├── data/                  # Dataset loaders and patch utilities
│   ├── models/                # CNN descriptor models and training logic
│   └── evaluation/            # Homography and metric evaluation
└── outputs/                   # Saved models, logs, and result figures
```

---

## Final Report Table of Contents

1. Title & Executive Summary  
2. Introduction & Problem Statement  
3. Background & Theory  
4. Dataset & Preprocessing  
5. Synthetic View Generation  
6. CNN Descriptor Architectures  
7. Training Setup  
8. Feature Matching + Homography Estimation  
9. SVD & Conditioning Analysis  
10. Results  
11. Discussion  
12. Conclusion  
13. References  
14. Appendix  

---

## Usage

### Setup

```bash
conda env create -f environment.yaml
conda activate regenv2
python build_scaffold.py
```

### Run Development Workflow

- Follow notebooks in `notebooks/` to execute each experiment phase.
- Use `src/` modules to extend or script additional evaluations.
- All outputs are saved to the `outputs/` directory.

---

## 🎓 Author

Brandon Henley  
CSCI E-25: Computer Vision, Harvard Extension School  
GitHub: [bchenley](https://github.com/bchenley)