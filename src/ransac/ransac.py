import random
import numpy as np

from tqdm import tqdm

def homography_matrix(P, m):
    A = []
    for r in range(len(P)):
        A.append([-P[r,0], -P[r,1], -1, 0, 0, 0, P[r,0]*m[r,0], P[r,1]*m[r,0], m[r,0]])
        A.append([0, 0, 0, -P[r,0], -P[r,1], -1, P[r,0]*m[r,1], P[r,1]*m[r,1], m[r,1]])

    u, s, vt = np.linalg.svd(A)
    H = np.reshape(vt[8], (3,3))
    H = (1/H.item(8)) * H # norm H33 to 1
    return H
    
def best_homography_RANSAC(good_matches):
    dst_pts = [] #destination image
    src_pts = [] #source image
    for ptA, ptB in good_matches:
        dst_pts.append(list(ptA)) 
        src_pts.append(list(ptB))
    dst_pts = np.array(dst_pts)
    src_pts = np.array(src_pts)
        
    num_sample = len(good_matches)
    threshold = 5
    iter = 500
    max_inlier = 0
    best_H = None
        
    for _ in tqdm(range(iter)):
        sub_sample_idx = random.sample(range(num_sample), 4)
        H = homography_matrix(src_pts[sub_sample_idx], dst_pts[sub_sample_idx])
        num_inlier = 0 

        for i in range(num_sample):
            if i not in sub_sample_idx:
                concateCoor = np.hstack((src_pts[i], [1]))
                dstCoor = H @ concateCoor.T
                if dstCoor[2] <= 1e-8:
                    continue
                dstCoor = dstCoor / dstCoor[2]
                if np.linalg.norm(dstCoor[:2] - dst_pts[i]) < threshold:
                    num_inlier = num_inlier + 1
        if max_inlier < num_inlier:
            max_inlier = num_inlier
            best_H = H
            print("inlier:", max_inlier)
    return best_H