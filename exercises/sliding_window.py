import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import exercises.draw_boxes

# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    span_x = x_start_stop[1] - x_start_stop[0]
    span_y = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    pixel_per_step_x = np.int(xy_window[0] * (1 - xy_overlap[0]))
    pixel_per_step_y = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    n_windows_x = np.int((span_x - (xy_window[0] * xy_overlap[0])) // pixel_per_step_x)
    n_windows_y = np.int((span_y - (xy_window[1] * xy_overlap[1])) // pixel_per_step_y)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    #     Note: you could vectorize this step, but in practice
    #     you'll be considering windows one by one with your
    #     classifier, so looping makes sense
    for win_y in range(n_windows_y):
        start_y = win_y * pixel_per_step_y + y_start_stop[0]
        y_height = start_y + xy_window[1]
        for win_x in range(n_windows_x):
            # Calculate each window position
            start_x = win_x * pixel_per_step_x + x_start_stop[0]
            x_width = start_x + xy_window[0]

            # Append window position to list
            window_list.append(((start_x, start_y), (x_width, y_height)))

    # Return the list of windows
    return window_list

"""
image = mpimg.imread('../test_images/bbox-example-image.jpg')
windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[400, None],
                    xy_window=(128, 128), xy_overlap=(0.5, 0.5))

window_img = draw_boxes.draw_boxes(image, windows, color=(0, 0, 255), thick=6)
plt.imshow(window_img)
plt.show()
"""
