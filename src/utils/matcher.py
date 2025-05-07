import cv2
import numpy as np

from tqdm import tqdm

def SIFT(left, right):
    SIFT_Detector = cv2.SIFT_create()
    kp1, des1 = SIFT_Detector.detectAndCompute(left, None)
    kp2, des2 = SIFT_Detector.detectAndCompute(right, None)
    return kp1, des1, kp2, des2

def knn_match(des1, des2, K):
    matches = []
    for i in tqdm(range(len(des1))):
        distances = np.linalg.norm(des1[i] - des2, axis=1)
        nearest_indices = np.argpartition(distances, K)[:K]
        best_match_idx = nearest_indices[0]
        second_best_match_idx = nearest_indices[1]
        best_distance = distances[best_match_idx]
        second_best_distance = distances[second_best_match_idx]
        matches.append([best_match_idx, best_distance, second_best_match_idx, second_best_distance])
    return matches

def lowe_ratio_test(matches, kp1, kp2, threshold):
    good_matches = []
    for i in range(len(matches)):
        if matches[i][1] < threshold * matches[i][3]:
            good_matches.append((i, matches[i][0]))

    good_matching = []
    for m,n in good_matches:
        ptA = (kp1[m].pt[0], kp1[m].pt[1])
        ptB = (kp2[n].pt[0], kp2[n].pt[1])
        good_matching.append([ptA, ptB])
    return good_matching