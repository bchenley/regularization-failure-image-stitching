# Author: Brandon Henley

import os
import cv2
import json
import random
import numpy as np
from pathlib import Path

def apply_affine_transform(image, shift_range=0.1, rotation_range=15, scale_range=(0.9, 1.1)):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    angle = random.uniform(-rotation_range, rotation_range)
    scale = random.uniform(*scale_range)
    tx = random.uniform(-shift_range, shift_range) * w
    ty = random.uniform(-shift_range, shift_range) * h

    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[:, 2] += (tx, ty)

    transformed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return transformed

def apply_occlusion(image, occlusion_prob=0.3):
    if random.random() > occlusion_prob:
        return image
    h, w = image.shape[:2]
    x = random.randint(0, w - w // 5)
    y = random.randint(0, h - h // 5)
    bw, bh = random.randint(w // 10, w // 5), random.randint(h // 10, h // 5)
    image[y:y+bh, x:x+bw] = 0
    return image

def apply_blur(image, blur_prob=0.3):
    if random.random() > blur_prob:
        return image
    ksize = random.choice([3, 5])
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def transform_image(image, shift_range, rotation_range, scale_range, occlusion_prob, blur_prob):
    image = apply_affine_transform(image, shift_range, rotation_range, scale_range)
    image = apply_occlusion(image, occlusion_prob)
    image = apply_blur(image, blur_prob)
    return image

def generate_pairs(
    input_dir,
    output_dir,
    manifest_path,
    n_samples=None,
    shift_range=0.1,
    rotation_range=15,
    scale_range=(0.9, 1.1),
    occlusion_prob=0.3,
    blur_prob=0.3
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    manifest_path = Path(manifest_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob("*.jpg"))
    if n_samples:
        files = random.sample(files, n_samples)

    manifest = []
    for idx, file_path in enumerate(files, start=1):
        img_id = f"{idx:06d}"
        img_orig_path = output_dir / f"img_{img_id}.jpg"
        img_warp_path = output_dir / f"img_{img_id}_warped.jpg"

        img = cv2.imread(str(file_path))
        if img is None:
            print(f"Could not read: {file_path}")
            continue

        img_warped = transform_image(
            img.copy(),
            shift_range, rotation_range, scale_range,
            occlusion_prob, blur_prob
        )

        cv2.imwrite(str(img_orig_path), img)
        cv2.imwrite(str(img_warp_path), img_warped)

        manifest.append({
            "id": img_id,
            "img1": str(img_orig_path),
            "img2": str(img_warp_path)
        })

        print(f"[{idx}/{len(files)}] Saved pair: {img_orig_path.name} <-> {img_warp_path.name}")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nAll done. {len(manifest)} image pairs created.")
    print(f"Manifest saved to {manifest_path.resolve()}")

if __name__ == "__main__":
    
    generate_pairs(
        input_dir="data/raw/selected",
        output_dir="data/processed",
        manifest_path="data/pairs.json",
        n_samples=None,
        shift_range=0.1,
        rotation_range=15,
        scale_range=(0.9, 1.1),
        occlusion_prob=0.3,
        blur_prob=0.3
    )