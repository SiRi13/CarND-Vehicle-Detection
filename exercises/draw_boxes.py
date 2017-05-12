import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def draw_boxes(img, bboxes, color=(255, 0, 0), thick=3):
    # make copy to draw on
    ret_img = np.copy(img)
    # iterate bboxes
    for box in bboxes:
        # draw box
        cv2.rectangle(ret_img, box[0], box[1], color, thick)

    return ret_img

"""
image = mpimg.imread('../test_images/bbox-example-image.jpg')
bboxes = [((850, 500), (1100, 675)), ((280, 500), (380, 570))]

result = draw_boxes(image, bboxes)
plt.imshow(result)
"""
