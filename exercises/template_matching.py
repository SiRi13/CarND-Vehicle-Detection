import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import draw_boxes

def find_matches(img, template_list):
    # Define an empty list to take bbox coords
    bbox_list = []
    # set match method
    method = cv2.TM_SQDIFF
    # Iterate through template list
    for template_name in template_list:
    # Read in templates one by one
        template = mpimg.imread(template_name)
        # Use cv2.matchTemplate() to search the image
        #     using whichever of the OpenCV search methods you prefer
        res = cv2.matchTemplate(img,template,method)
        # Use cv2.minMaxLoc() to extract the location of the best match
        _, _, min_loc, max_loc = cv2.minMaxLoc(res)
        h, w, _ = template.shape
        # Determine bounding box corners for the match
        if method in  [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right = (top_left[0] + w, top_left[1] + h)
        bbox_list.append((top_left, bottom_right))
    # Return the list of bounding boxes
    return bbox_list


image = mpimg.imread('../test_images/bbox-example-image.jpg')
templist = ['../test_images/cutout1.jpg', '../test_images/cutout2.jpg',
            '../test_images/cutout3.jpg', '../test_images/cutout4.jpg',
            '../test_images/cutout5.jpg', '../test_images/cutout6.jpg']

bboxes = find_matches(image, templist)
result = draw_boxes.draw_boxes(image, bboxes)
plt.imshow(result)
