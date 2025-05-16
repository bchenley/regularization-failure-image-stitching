# Author: Brandon Henley

import numpy as np

def build_dlt_matrix(pts1, pts2):
    assert pts1.shape == pts2.shape
    N = pts1.shape[0]
    A = []
    for i in range(N):
        x, y = pts1[i]
        x_p, y_p = pts2[i]
        A.append([0, 0, 0, -x, -y, -1, y_p * x, y_p * y, y_p])
        A.append([x, y, 1, 0, 0, 0, -x_p * x, -x_p * y, -x_p])
    return np.array(A)

def estimate_homography_dlt(matches):
    pts1 = np.float32([m[0] for m in matches])
    pts2 = np.float32([m[1] for m in matches])
    A = build_dlt_matrix(pts1, pts2)
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1]
    H = h.reshape(3, 3)
    return H / H[2, 2], S