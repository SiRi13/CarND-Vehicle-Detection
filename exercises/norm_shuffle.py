import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.preprocessing import StandardScaler

import utils
import color_hist
import spatial_bin
import get_hog

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32), hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0, hist_range=(0, 256),
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = list()
    # Iterate through the list of images
    for img in imgs:
        img_features = list()
        # Read in each one by one
        image = mpimg.imread(img)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            cv2_space = eval('cv2.COLOR_RGB2' + color_space)
            feature_image = cv2.cvtColor(image, cv2_space)
        else:
            feature_image = np.copy(image)
        # Apply bin_spatial() to get spatial color features
        if spatial_feat:
            spatial_features = spatial_bin.bin_spatial(feature_image, size=spatial_size)
            img_features.append(spatial_features)
        # Apply color_hist() to get color histogram features
        if hist_feat:
            _, _, _, _, hist_features = color_hist.color_hist(feature_image, nbins=hist_bins)
            img_features.append(hist_features)
        # Apply  get_hog_features() with vis=False, feature_vec=True
        if hog_feat == True:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(image.shape[2]):
                    hog_features.append(get_hog.get_hog_features(feature_image[:,:,channel],
                                        orient, pix_per_cell, cell_per_block, False, True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog.get_hog_features(feature_image[:,:,hog_channel], orient,
                            pix_per_cell, cell_per_block, False, True)
            # Append the new feature vector to the features list
            img_features.append(hog_features)
        # Append the new feature vector to the features list
        features.append(np.concatenate(img_features))
    # Return list of feature vectors
    return features

"""
cars, notcars = utils.get_images()

car_features = extract_features(cars[:1], color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256))
notcar_features = extract_features(notcars[:1], color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256))

if len(car_features) > 0:
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    car_ind = np.random.randint(0, len(car_features))
    # Plot an example of raw and scaled features
    fig = plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.imshow(mpimg.imread(cars[car_ind]))
    plt.title('Original Image')
    plt.subplot(132)
    plt.plot(X[car_ind])
    plt.title('Raw Features')
    plt.subplot(133)
    plt.plot(scaled_X[car_ind])
    plt.title('Normalized Features')
    fig.tight_layout()
    plt.show()
"""
