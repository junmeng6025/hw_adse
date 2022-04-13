"""
Goal of Task 2:
    Extract SIFT features from the images to enable image matching.

Hint: Execute feature_matcher.py to validate your implementation.
"""


import cv2


def detect_features(img_name):
    """
    Detects and computes features and their descriptors.

    input:
        img_name (type: str): path to given image

    outputs:
        img (type: np.ndarray): opened image
        kp (type: list): keypoints
        des (type: np.ndarray): descriptors
    """

    # Task:
    # ToDo: Implement the following steps:
    #   1. create SIFT detector (cv2)
    #   2. read img_name (cv2)
    #   3. calculate keypoints and descriptors
    ########################
    #  Start of your code  #
    ########################

    img = cv2.imread(img_name)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)

    ########################
    #   End of your code   #
    ########################

    return img, kp, des
