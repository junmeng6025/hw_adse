"""
Goal of Task 2:
    Implement a helper function to transform the label format into the required format in YOLOv3.
"""


import numpy as np


def xywh2xyxy_np(xywh):
    """
    input:
        xywh (type: np.ndarray, shape: (n,4), dtype: int16): n bounding boxes with the xywh format (center based)

    output:
        xyxy (type: np.ndarray, shape: (n,4), dtype: int16): n bounding boxes with the xyxy format (edge based)
    """

    # Task:
    # ToDo: Implement the conversion of all n bounding boxes from xywh format (center x, center y, width w, height h)
    #   to the xyxy format (top left corner x, top left corner y, bottom right corner x, bottom right corner y).
    # Hints:
    #   - All values are in pixel, ranging from 0 to 352 (exclusive).
    #   - The first dimension of xywh is n (number of input boxes) and not 1.
    #   - The dtype of the resulting array is int16. Floor the xyxy pixel values to the next integer AFTER
    #     the calculation (example: xywh = np.asarray([1, 1, 1, 1]) transforms to xyxy = np.asarray([0, 0, 1, 1])).
    ########################
    #  Start of your code  #
    ########################

    xyxy = np.zeros_like(xywh)
    xyxy[:, 0] = np.floor(xywh[:, 0] - 0.5 * xywh[:, 2])
    xyxy[:, 1] = np.floor(xywh[:, 1] - 0.5 * xywh[:, 3])
    xyxy[:, 2] = np.floor(xywh[:, 0] + 0.5 * xywh[:, 2])
    xyxy[:, 3] = np.floor(xywh[:, 1] + 0.5 * xywh[:, 3])

    ########################
    #   End of your code   #
    ########################

    return xyxy


if __name__ == "__main__":
    # Execute this file to check your output of this example
    xywh_example = np.asarray([[150, 120, 20, 10], [258, 89, 55, 45]], dtype=np.int16)
    your_xyxy = xywh2xyxy_np(xywh_example)
    print(f"Your xyxy: {your_xyxy}")
    print(f"Your xyxy shape: {your_xyxy.shape}")
    print(f"Your xyxy dtype: {your_xyxy.dtype}")
