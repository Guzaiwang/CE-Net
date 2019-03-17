import os
import scipy.misc as misc
import shutil
import cv2
import Constants
import numpy as np

from skimage import morphology


def extract_each_layer(image, threshold):
    """
    This image processing funtion is designed for the OCT image post processing.
    It can remove the small regions and find the OCT layer boundary under the specified threshold.
    :param image:
    :param threshold:
    :return:
    """
    # convert the output to the binary image
    ret, binary = cv2.threshold(image, threshold, 1, cv2.THRESH_BINARY)

    bool_binary = np.array(binary, bool)

    # remove the small object
    remove_binary = morphology.remove_small_objects(bool_binary, min_size=25000,
                                                                connectivity=2,
                                                                in_place=False)
    c = np.multiply(bool_binary, remove_binary)
    final_binary = np.zeros(shape=np.shape(binary))
    final_binary[c == True] = 1
    binary_image = cv2.filter2D(final_binary, -1, np.array([[-1], [1]]))
    layer_one = np.zeros(shape=[1, np.shape(binary_image)[1]])
    for i in range(np.shape(binary_image)[1]):
        location_point = np.where(binary_image[:, i] > 0)[0]
        # print(location_point)

        if len(location_point) == 1:
            layer_one[0, i] = location_point
        elif len(location_point) == 0:
            layer_one[0, i] = layer_one[0, i-1]

        else:
            layer_one[0, i] = location_point[0]

    return layer_one


if __name__ == '__main__':
    image_path = '/home/jimmyliu/Zaiwang/crop-OCT/train/562.fds/crop-images/' \
                 'oct202.png'
    gt_path = '/home/jimmyliu/Zaiwang/crop-OCT/train/562.fds/crop-gt/' \
                 'oct202.png'
    image = cv2.imread(image_path)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('gt.png', gt)
    print(np.max(image), np.shape(image))
    print(np.max(gt), np.shape(gt))