import os
import cv2
import collections
import numpy as np
import matplotlib.image as mpimg
from scipy.ndimage.measurements import label

class VehicleFilter:

    def __init__(self, thresh=1.0, hist_len=10, export=False):
        self.threshold = thresh
        self.history = collections.deque(maxlen=hist_len+1)
        self.export = export

    def add_heat(self, heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heatmap# Iterate through list of bboxes

    def apply_threshold(self, heatmap):
        # Zero out pixels below the threshold
        heatmap[heatmap <= self.threshold] = 0
        # Return thresholded map
        return heatmap

    def draw_labeled_bboxes(self, img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1]):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0, 255, 0), 6)
        # Return the image
        return img

    def do_export(self, images, filename, color_map='gray'):
        if self.export == True:
            fig = plt.figure()
            for idx in range(1, len(images)+1):
                plt.subplot(2,3,idx)
                plt.imshow(images[idx])
            fig.tight_layout()
            plt.savefig(os.path.join('./output_images/', filename), format='png')

    def filter(self, frame, draw_frame, windows, reset):
        if reset == True:
            self.history.clear()

        self.history.append(windows)

        heatmap = np.zeros_like(frame[:, :, 0], dtype=np.float)
        for bboxes in self.history:
            heatmap = self.add_heat(heatmap, bboxes)

        thresh_heatmap = self.apply_threshold(heatmap)

        labels = label(thresh_heatmap)

        ret_img = self.draw_labeled_bboxes(draw_frame, labels)

        imgs_to_export = [frame, draw_frame, heatmap, thresh_heatmap, labels, ret_img]
        self.do_export(imgs_to_export, 'filter_combined.png')

        return ret_img
