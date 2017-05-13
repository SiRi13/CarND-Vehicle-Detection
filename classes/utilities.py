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
                    transform_sqrt=True, visualise=vis, feature_vector=feature_vec)

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
