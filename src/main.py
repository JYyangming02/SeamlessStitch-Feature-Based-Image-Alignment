import cv2

from utils.image_io import read_img, img_to_gray
from utils.matcher import SIFT, knn_match, lowe_ratio_test
from ransac.ransac import best_homography_RANSAC
from blending.blending import Blending


def main(img1, img2):
    # SIFT
    left_img = img_to_gray(img1)
    right_img = img_to_gray(img2)
    kp1, des1, kp2, des2 = SIFT(left_img, right_img)

    # Knn
    matches = knn_match(des1, des2, K=2)
    good_matching = lowe_ratio_test(matches, kp1, kp2, threshold=0.75)

    # RANSAC to find best-fit homography matrix
    H = best_homography_RANSAC(good_matching)

    # stitching
    blending = Blending(left_image=img1, right_image=img2)
    result = blending.blending(H)
    #result = blending.single_blending(H)
    cv2.imwrite("test.jpg", result)
    
if __name__ == '__main__':
    # the example of image window
    # creat_im_window("Result",img)
    # im_show()

    # you can use this function to store the result
    # cv2.imwrite("result.jpg",img)

    img1, img1_gray = read_img("datasets/Base/Base1.jpg")
    img2, img2_gray = read_img("datasets/Base/Base2.jpg")
    main(img1 ,img2)