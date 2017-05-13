import cv2
import utilities as utils

class VehicleDetector:

    def __init__(self, clf, sclr, values):
        self.classifier = clf
        self.scalar = sclr

        self.__init_values__(values)

    def __init_values(self, values):
        self.color_space = values['color_space']
        # HoG params
        self.hog_feat = values['hog_feat']
        self.hog_features = list()
        self.orientations = values['orientations']
        self.pix_per_cell = values['pix_per_cell']
        self.cell_per_block = values['cell_per_block']
        self.hog_channel = values['hog_channel']
        # Spatial Binning params
        self.spatial_feat = values['spatial_feat']
        self.spatial_size = values['spatial_size']
        # Histogram params
        self.hist_feat = values['hist_feat']
        self.hist_bins = values['hist_bins']

    def compute_hog_features(self, image):
        if self.hog_channel == 'ALL':
            for channel in range(feature_image.shape[2]):
                self.hog_features.extend(utils.get_hog_features(feature_image[:,:,channel], self.orientations, self.pix_per_cell, self.cell_per_block,
                                                                vis=False, feature_vec=False))
        else:
            self.hog_features = utils.get_hog_features(feature_image[:, :, self.hog_channel], self.orient, self.pix_per_cell, self.cell_per_block,
                                                        vis=False, feature_vec=False)

    def single_img_features(self, img):
        #1) Define an empty list to receive features
        img_features = []
        #2) Apply color conversion if other than 'RGB'
        feature_image = utils.convert_or_copy(img, self.color_space)
        #3) Compute spatial features if flag is set
        if self.spatial_feat == True:
            spatial_features = utils.bin_spatial(feature_image, size=self.spatial_size)
            #4) Append features to list
            img_features.append(spatial_features)
        #5) Compute histogram features if flag is set
        if self.hist_feat == True:
            hist_features = utils.color_hist(feature_image, nbins=self.hist_bins)
            #6) Append features to list
            img_features.append(hist_features)
        #7) Compute HOG features if flag is set
        if self.hog_feat == True:
            #8) Append features to list
            img_features.append(self.hog_features)

        #9) Return concatenated array of features
        return np.concatenate(img_features)

    # Define a function you will pass an image
    # and the list of windows to be searched (output of slide_windows())
    def search_windows(self, img, windows):
        self.compute_hog_features(img)
        #1) Create an empty list to receive positive detection windows
        on_windows = []
        #2) Iterate over all windows in the list
        for window in windows:
            #3) Extract the test window from original image
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            #4) Extract features for that window using single_img_features()
            features = self.single_img_features(test_img)
            #5) Scale extracted features to be fed to classifier
            test_features = self.scaler.transform(np.array(features).reshape(1, -1))
            #6) Predict using your classifier
            prediction = self.classifier.predict(test_features)
            #7) If positive (prediction == 1) then save the window
            if prediction == 1:
                on_windows.append(window)
        #8) Return windows for positive detections
        return on_windows
