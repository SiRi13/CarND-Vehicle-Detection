import cv2
import numpy as np
import classes.utilities as utils

class VehicleDetector:

    def __init__(self, clf, sclr, values, fltr):
        self.classifier = clf
        self.scaler = sclr
        self.filter = fltr
        self.__init_values__(values)

    def __init_values__(self, values):
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
        # sliding windows
        self.windows = None
        self.frame_size = None

    def compute_hog_features(self, img):
        if self.hog_channel == 'ALL':
            hog_features = list()
            for channel in range(img.shape[2]):
                hog_features.append(utils.get_hog_features(img[:,:,channel], self.orientations, self.pix_per_cell, self.cell_per_block,
                                                                vis=False, feature_vec=False))
            self.hog_features = np.array(hog_features)
        else:
            self.hog_features = utils.get_hog_features(img[:, :, self.hog_channel], self.orient, self.pix_per_cell, self.cell_per_block,
                                                        vis=False, feature_vec=False)

    def single_img_features(self, img, hog_window, separately=False):
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
            hog_start, hog_width = hog_window
            start_x = hog_start[1]
            x_width = hog_start[1] + hog_width
            start_y = hog_start[0]
            y_height = hog_start[0] + hog_width
            img_features.append(self.hog_features[:, start_x:x_width, start_y:y_height, :, :, :].ravel())

        if separately == True:
            return img_features

        #9) Return concatenated array of features
        return np.concatenate(img_features)

    # Define a function you will pass an image
    # and the list of windows to be searched (output of slide_windows())
    def search_windows(self, img):
        self.compute_hog_features(img)
        #1) Create an empty list to receive positive detection windows
        on_windows = []
        #2) Iterate over all windows in the list
        for window in self.windows:
            #3) Extract the test window from original image
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            hog_x = (window[0][0] // self.pix_per_cell, window[0][1] // self.pix_per_cell)
            x_width = (img.shape[1] // self.pix_per_cell) - 1
            #4) Extract features for that window using single_img_features()
            features = self.single_img_features(test_img, (hog_x, x_width))
            #5) Scale extracted features to be fed to classifier
            test_features = self.scaler.transform(np.array(features).reshape(1, -1))
            #6) Predict using your classifier
            prediction = self.classifier.predict(test_features)
            #7) If positive (prediction == 1) then save the window
            if prediction == 1:
                on_windows.append(window)
        #8) Return windows for positive detections
        return on_windows

    def get_sliding_windows(self, frame, separately=False):
        window_settings = [([None, None], (360, 504), (64, 64), (0.8, 0.8)),
                            ([None, None], (360, 576), (96, 96), (0.6, 0.6)),
                            ([None, None], (360, 648), (128, 128), (0.5, 0.5))]

        windows = list()
        for settings in window_settings:
            slide_windows = utils.slide_window(frame.shape, x_start_stop=settings[0], y_start_stop=settings[1], xy_window=settings[2], xy_overlap=settings[3])
            if separately == True:
                windows.append(slide_windows)
            else:
                windows += slide_windows

        self.frame_size = frame.shape
        return windows

    def process_frame(self, frame, reset=False):
        if self.windows is None or (self.frame_size != frame.shape):
            self.windows = self.get_sliding_windows(frame)
        # copy frame
        draw_frame = np.copy(frame)
        # scale image
        frame = frame.astype(np.float32) / 255
        # compute hog features once
        self.compute_hog_features(frame)
        # search for vehicles
        hot_windows = self.search_windows(frame)
        # filter false-positives and multiple detections
        draw_frame = self.filter.filter(frame, draw_frame, hot_windows, reset)

        return draw_frame
