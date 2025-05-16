# When Regularization Fails: A Study in Image Stitching

This project investigates the unintended effects of regularization methods (L2, Dropout) on CNN-based local feature descriptor models in keypoint-based homography estimation and image stitching. The goal is to identify conditions under which regularization degrades descriptor performance, resulting in poor geometric alignment or unstable homographies.

---

## 📂 Project Structure

```
project/
├── build_scaffold.py          # Creates all necessary folders
├── environment.yaml           # Conda environment definition
├── README.md                  # Project overview and usage
├── notebooks/                 # Main development notebook
├── data/                      # COCO source + generated image pairs
│   ├── raw/                   # Raw training/evaluation images
│   ├── processed/             # Transformed image pairs
│   └── pairs_train.json       # Mapping original to warped views
├── scripts/                   # One-off helpers (augmentation, manifest gen)
├── src/                       # Core modular Python code
│   ├── data/                  # Dataset loading and patch sampling
│   ├── models/                # CNN descriptor models and training
│   ├── evaluation/            # Matching, homography, and metric logging
└── outputs/                   # Trained models, metrics, and logs
```

---

## 🧭 Project Procedure: Effects of Regularization on Descriptor Stability in Homography Estimation

This notebook presents a complete experimental pipeline to study how different regularization strategies (L2, Dropout, Hybrid) affect CNN-based keypoint descriptors used in homography estimation and geometric alignment. The project is framed around the inverse problem posed by the DLT algorithm and evaluates both spatial precision and numerical conditioning.

---

## Phase 1: Dataset Preparation

- 1.1 Unzip the COCO 2014 image dataset if not already extracted.
- 1.2 Randomly select a subset of images.
    - **1000 images** from `train2014/` and copy them into `data/raw/selected/` for training.
    - **200 images** for held-out evaluation and save to `data/raw/eval/`.
    - Save training and test filenames to `train_list.txt` and `eval_list.txt`.

---

## Phase 2: Synthetic View Generation
- Apply transformations: rotation, scale, occlusion, blur, and mild perspective.
- Save the original and transformed image pair to `data/processed/train/` or `data/processed/eval/`.
- Generate `pairs_train.json` and `pairs_eval.json` manifests mapping each original image to its warped counterpart.

---

## Phase 3: Descriptor Training with Regularization

- 3.1 Define a CNN descriptor model with tunable architecture.
- 3.2 Model training:
    - 3.2.a. **Baseline** (no regularization)
    - 3.2.b. **L2-Regularized**
    - 3.2.c. **Dropout-Regularized**
    - 3.2.d. **Hybrid** (L2 + Dropout)
- 3.3 Track training and validation loss.

*Note*: 
- Early stopping was mentioned in the proposal, but was not used in the report. 
- SSIM was mentioned in the proposal, but was excluded in the report due to alignment artifacts.
- Hyperparameter tuning was originally planned but skipped due to time constraints.

---

## Phase 4: Single-Pair Evaluation

- 4.1 Select a representative test pair.
- 4.2 Detect Shi-Tomasi corners and extract patch descriptors.
- 4.3 Match descriptors using NN + Lowe’s ratio test.
- 4.4 Estimate homography matrix **H** using DLT with SVD.
- 4.5 Homography Analysis:
    - Singular values
    - Condition number κ(A)
    - **RMSE** and **SSIM** for perceptual and geometric alignment.

---

## Phase 5: Evaluation on Held-Out Image Pairs

- Evaluate each model variant on all 200 evaluation images.
- Compute per-pair metrics:
    - Match count
    - Condition number κ(A)
    - RMSE (pixel alignment error)

---

## Phase 6: Metric Analysis and Visualization

- 6.1 Results
- 6.2 Discussion

---

## Phase 7: Final Interpretation and Write-Up

- 7.1 Hypothesis and empirical findings.
- 7.2 Effect of regularization on descriptor robustness, matchability, and alignment.
- 7.3 Numerical stability vs. spatial precision.
- 7.4 Limitations

---

## Phase 8: References and Links

- 8.1 References
- 8.2 Links

---

## 🛠 Usage

### Setup

```bash
conda env create -f environment.yaml
conda activate regenv
python build_scaffold.py
```

### Run Development Workflow

- Follow `notebooks/reg_failure_stitching.ipynb` to run the complete pipeline.
- Output logs and trained models will be stored in `outputs/`.

---

## 🎓 Author

Brandon Henley  
CSCI E-25: Computer Vision, Harvard Extension School