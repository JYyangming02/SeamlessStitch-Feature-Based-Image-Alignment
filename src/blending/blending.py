import cv2
import numpy as np

class Blending:
    def __init__(self, left_image, right_image):
        self.left_image = left_image
        self.right_image = right_image
        self.height_img_l = self.left_image.shape[0]
        self.width_img_l = self.left_image.shape[1]
        self.width_img_r = self.right_image.shape[1]
        self.height_panorama = self.height_img_l
        self.width_panorama = self.width_img_l + self.width_img_r

    def create_mask(self, version):
        offset = int(800 / 2)
        barrier = self.left_image.shape[1] - int(800 / 2)
        mask = np.zeros((self.height_panorama, self.width_panorama))

        if version == 'left_image':
            mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(1, 0, 2 * offset).T,
                                                                  (self.height_panorama, 1))
            mask[:, :barrier - offset] = 1
        else:
            mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(0, 1, 2 * offset).T,
                                                                  (self.height_panorama, 1))
            mask[:, barrier + offset:] = 1
        return cv2.merge([mask, mask, mask])

    def blending(self, H):
        panorama1 = np.zeros((self.height_panorama, self.width_panorama, 3))
        mask1 = self.create_mask(version='left_image')

        panorama1[0:self.left_image.shape[0], 0:self.left_image.shape[1], :] = self.left_image
        panorama1 *= mask1

        mask2 = self.create_mask(version='right_image')
        panorama2 = cv2.warpPerspective(self.right_image, H, (self.width_panorama, self.height_panorama))*mask2

        result = panorama1 + panorama2

        rows, cols = np.where(result[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1
        final_result = result[min_row:max_row, min_col:max_col, :]
        return final_result
    
    def single_blending(self, H):
        # Find left image corner
        height_l, width_l, channel_l = self.left_image.shape
        corners = [[0, 0, 1], # Top-left
                [width_l, 0, 1], # Top-right
                [width_l, height_l, 1], # Bottom-Right
                [0, height_l, 1]] # Bottom-left
        
        corners_new = np.dot(corners, H)
        corners_new = np.array(corners_new).T
        corners_x = corners_new[0] / corners_new[2]
        corners_y = corners_new[1] / corners_new[2]
        x_min = min(min(corners_x), 0)
        y_min = min(min(corners_y), 0)

        height_new = int(round(height_l + abs(y_min)))
        width_new = int(round(width_l + abs(x_min) ))
        size = (width_new, height_new)
 
        # translation matrix
        translation_mat = np.array([[1, 0, -x_min], 
                                    [0, 1, -y_min], 
                                    [0, 0, 1]])
        H = np.dot(translation_mat, H)
        
        panorama1 = cv2.warpPerspective(self.left_image, H, size)
        panorama2 = cv2.warpPerspective(self.right_image, translation_mat, size)
        
        final_result = np.concatenate((panorama1, panorama2), axis=1)
        return final_result  