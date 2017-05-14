import cv2
import time
import pickle
import random
import numpy as np
import matplotlib.image as mpimg

import classes.utilities as utils

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.utils import shuffle as skShuffle

class ClassifierTrainer:

    def __init__(self, path='./test_images/{}/**/*.png'):
        self.cars, self.notcars = utils.load_images(path)

        self.__init_values__()

    def __init_values__(self):
        self.trained = False
        self.color_space = cv2.COLOR_RGB2YCrCb # possible values RGB, HSV, LUV, HLS, YUV, YCrCb
        # HoG params
        self.hog_feat = True
        self.orientations = 8
        self.pix_per_cell = 8
        self.cell_per_block = 2
        self.hog_channel = "ALL" # possible values 0, 1, 2 or "ALL"
        # Spatial Binning params
        self.spatial_feat = True
        self.spatial_size = (32, 32)
        # Histogram params
        self.hist_feat = True
        self.hist_bins = 16

    def export_settings(self):
        if self.trained == True:
            to_persist = dict()
            to_persist['color_space'] = self.color_space
            to_persist['hog_feat'] = self.hog_feat
            to_persist['orientations'] = self.orientations
            to_persist['pix_per_cell'] = self.color_space
            to_persist['cell_per_block'] = self.cell_per_block
            to_persist['hog_channel'] = self.hog_channel
            to_persist['spatial_feat'] = self.spatial_feat
            to_persist['spatial_size'] = self.spatial_size
            to_persist['hist_feat'] = self.hist_feat
            to_persist['hist_bins'] = self.hist_bins
            joblib.dump(to_persist, './classes/settings/parameters.p')

            to_persist = dict()
            to_persist['classifier'] = self.svc
            to_persist['scaler'] = self.X_scaler
            joblib.dump(to_persist, './classes/settings/svc.p')

    def extract_features(self, img_paths):
        # Create a list to append feature vectors to
        features = list()
        # Iterate through the list of images
        for idx, img in enumerate(img_paths):
            img_features = list()
            # Read in each one by one
            image = mpimg.imread(img)
            # apply color conversion if other than 'RGB'
            feature_image = utils.convert_or_copy(image, self.color_space)
            # Apply bin_spatial() to get spatial color features
            if self.spatial_feat == True:
                spatial_features = utils.bin_spatial(feature_image, size=self.spatial_size)
                img_features.append(spatial_features)
            # Apply color_hist() to get color histogram features
            if self.hist_feat == True:
                hist_features = utils.color_hist(feature_image, nbins=self.hist_bins)
                img_features.append(hist_features)
            # Apply  get_hog_features() with vis=False, feature_vec=True
            if self.hog_feat == True:
                if self.hog_channel == 'ALL':
                    hog_features = []
                    for channel in range(image.shape[2]):
                        hog_features.append(utils.get_hog_features(feature_image[:, :, channel], self.orientations, self.pix_per_cell,
                                                                    self.cell_per_block, transform_sqrt= True, vis=False, feature_vec=True))
                    hog_features = np.ravel(hog_features)
                else:
                    hog_features = utils.get_hog_features(feature_image[:,:,hog_channel], self.orientations, self.pix_per_cell,
                                                            self.cell_per_block, transform_sqrt=False, vis=False, feature_vec=True)
                # Append the new feature vector to the features list
                img_features.append(hog_features)
            # Append the new feature vector to the features list
            features.append(np.concatenate(img_features))

            if idx % (len(img_paths) // 10) == 0:
                print(idx, ' of ', len(img_paths))
        # Return list of feature vectors
        return features

    def train(self):
        car_features = self.extract_features(self.cars)
        notcar_features = self.extract_features(self.notcars)

        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler
        self.X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        self.scaled_X = self.X_scaler.transform(X)
        # Define the labels vector
        self.y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # shuffle sets
        scaled_X, y = skShuffle(self.scaled_X, self.y)
        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(self.scaled_X, self.y, test_size=0.2, random_state=rand_state)

        print('Using:', self.orientations ,'orientations', self.pix_per_cell, 'pixels per cell and', self.cell_per_block,'cells per block')
        print('Feature vector length:', len(X_train[0]))
        # Use a linear SVC
        self.svc = LinearSVC()
        # Check the training time for the SVC
        t=time.time()
        self.svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(self.svc.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t=time.time()
        self.trained = True
