import cv2
import numpy as np
from skimage.feature import hog as skHog

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    return skHog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block),
                    transform_sqrt=True, visualise=vis, feature_vector=feature_vec)

def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    return cv2.resize(feature_image, size).ravel()

def color_hist(img, nbins=32, bins_range=(0.0, 1.0)):
    # Compute the histogram of the RGB channels separately
    ch0_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    ch1_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    ch2_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((ch0_hist, ch1_hist, ch2_hist))
    # return features
    return hist_features
