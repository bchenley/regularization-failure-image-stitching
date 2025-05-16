# Author: Brandon Henley
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from src.models.cnn_descriptor import CNNDescriptor
from src.data.patch_utils import extract_descriptors
from src.evaluation.matching import match_descriptors
from src.evaluation.homography import estimate_homography_dlt

def evaluate_model_on_pair(model, pair_id, model_name, patch_size=128, ratio_thresh=0.8, device = "cpu" ):
    """
    Evaluate a trained descriptor model on a single image pair.
    
    Returns:
        A dictionary with evaluation metrics for aggregation.
    """
    # Load images
    img1_path = Path(f"data/processed/img_{pair_id}.jpg")
    img2_path = Path(f"data/processed/img_{pair_id}_warped.jpg")
    img1 = np.array(Image.open(img1_path).convert("RGB"))
    img2 = np.array(Image.open(img2_path).convert("RGB"))

    # Convert to grayscale for keypoint detection
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # Detect Shi-Tomasi keypoints
    keypoints1 = cv2.goodFeaturesToTrack(gray1, maxCorners=500, qualityLevel=0.01, minDistance=8)
    keypoints2 = cv2.goodFeaturesToTrack(gray2, maxCorners=500, qualityLevel=0.01, minDistance=8)
    keypoints1 = keypoints1.squeeze(1)
    keypoints2 = keypoints2.squeeze(1)

    # Extract descriptors
    keypoints1_valid, desc1 = extract_descriptors(img1, keypoints1, model, device=device)
    keypoints2_valid, desc2 = extract_descriptors(img2, keypoints2, model, device=device)

    if len(desc1) < 4 or len(desc2) < 4:
        return {
            "pair_id": pair_id,
            "model": model_name,
            "match_count": 0,
            "kappa": np.nan,
            "rmse": np.nan
        }

    # Match descriptors
    matches = match_descriptors(desc1, desc2, keypoints1_valid, keypoints2_valid, ratio_thresh)
    if len(matches) < 4:
        return {
            "pair_id": pair_id,
            "model": model_name,
            "match_count": len(matches),
            "kappa": np.nan,
            "rmse": np.nan
        }

    # Estimate homography and singular values
    try:
        H, singular_values = estimate_homography_dlt(matches)
        sigma_max = singular_values[0]
        sigma_min = singular_values[-1]
        kappa = sigma_max / sigma_min
    except:
        return {
            "pair_id": pair_id,
            "model": model_name,
            "match_count": len(matches),
            "kappa": np.nan,
            "rmse": np.nan
        }

    # Warp and compute RMSE
    warped_img2 = cv2.warpPerspective(img2, H, dsize=(img1.shape[1], img1.shape[0]))
    rmse = np.sqrt(np.mean((img1.astype("float32") - warped_img2.astype("float32")) ** 2))

    return {
        "pair_id": pair_id,
        "model": model_name,
        "match_count": len(matches),
        "kappa": kappa,
        "rmse": rmse
    }