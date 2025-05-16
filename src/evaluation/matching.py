from scipy.spatial import cKDTree
import numpy as np

def match_descriptors(desc1, desc2, keypoints1, keypoints2, ratio_thresh=0.8):
    tree = cKDTree(desc2.numpy())
    dists, idxs = tree.query(desc1.numpy(), k=2)

    matches = []
    for i, (dist1, dist2) in enumerate(dists):
        if dist1 < ratio_thresh * dist2:
            matched_idx = idxs[i, 0]
            matches.append((keypoints1[i], keypoints2[matched_idx]))

    return np.array(matches)