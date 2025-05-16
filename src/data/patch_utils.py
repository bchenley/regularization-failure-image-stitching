import torch
import numpy as np
from torchvision import transforms
import cv2
import random

def extract_descriptors(img, keypoints, model, device="cpu", patch_size=64):
    
    descriptors = []
    valid_keypoints = []
    half_patch = patch_size // 2
    h, w, _ = img.shape

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((patch_size, patch_size))
    ])

    for (x, y) in keypoints:
        x, y = int(x), int(y)
        if x - half_patch < 0 or x + half_patch > w or y - half_patch < 0 or y + half_patch > h:
            continue
        patch = img[y - half_patch:y + half_patch, x - half_patch:x + half_patch, :]
        patch_tensor = preprocess(patch).unsqueeze(0).to(device)
        with torch.no_grad():
            desc = model(patch_tensor).cpu().squeeze()
        descriptors.append(desc)
        valid_keypoints.append([x, y])

    return np.array(valid_keypoints), torch.stack(descriptors)

def draw_matches(img1, img2, matches, max_draw=100):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:] = img2

    num_matches = min(max_draw, len(matches))
    matches = matches[:num_matches]

    for pt1, pt2 in matches:
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]), int(pt2[1])
        color = tuple([random.randint(0, 255) for _ in range(3)])
        pt1_canvas = (x1, y1)
        pt2_canvas = (x2 + w1, y2)
        cv2.circle(canvas, pt1_canvas, 3, color, -1)
        cv2.circle(canvas, pt2_canvas, 3, color, -1)
        cv2.line(canvas, pt1_canvas, pt2_canvas, color, 1)

    return canvas