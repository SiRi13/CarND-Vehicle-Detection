import cv2
import glob
import numpy as np

from skimage.feature import hog as skHog

def convert_or_copy(image, color_space):
    if color_space is not None:
        feature_image = cv2.cvtColor(image, color_space)
    else:
        feature_image = np.copy(image)

    if np.max(feature_image) > 1.0:
        feature_image = feature_image.astype(np.float32)/255

    return feature_image

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    return skHog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block),
                    transform_sqrt=True, visualise=True, feature_vector=feature_vec)

def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    return cv2.resize(img, size).ravel()

def color_hist(img, nbins=32, bins_range=(0.0, 1.0)):
    # Compute the histogram of the RGB channels separately
    ch0_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    ch1_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    ch2_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((ch0_hist[0], ch1_hist[0], ch2_hist[0]))
    # return features
    return hist_features

def load_images(glob_path):
    non_cars_images = glob.glob(glob_path.format('non-vehicles'))
    cars_images = glob.glob(glob_path.format('vehicles'))
    cars = []
    notcars = []

    for image_path in non_cars_images:
        notcars.append(image_path)

    for image_path in cars_images:
        cars.append(image_path)

    return cars, notcars

def slide_window(image_size, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = image_size[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = image_size[0]
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
