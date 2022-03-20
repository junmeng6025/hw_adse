import cv2
import matplotlib.pyplot as plt
import os

from task2 import detect_features


image_path = 'data/Mount_Rushmore'
image_name1 = os.path.join(image_path, 'image1.jpg')
image_name2 = os.path.join(image_path, 'image2.jpg')


def draw_image_matches(detector, img1_name, img2_name, nmatches=10):
    """
    Draw ORB feature matches of the given two images.
    """

    img1, kp1, des1 = image_detect_and_compute(detector, img1_name)
    img2, kp2, des2 = image_detect_and_compute(detector, img2_name)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)  # Sort matches by distance.  Best come first.
    
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:nmatches], img2, flags=2)  # Show top 10 matches
    plt.figure()
    plt.title(type(detector))
    plt.imshow(img_matches)
    plt.show()
    

img1, kp1, des1 = detect_features(image_name1)
img2, kp2, des2 = detect_features(image_name2)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

matchesMask = [[0, 0] for i in range(len(matches))]
# ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.5 * n.distance:
        matchesMask[i] = [1, 0]

draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask,
                   flags=0)

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
plt.figure()
plt.imshow(img3)
plt.show()
