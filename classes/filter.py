import os
import cv2
import collections
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
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
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
        # Return the image
        return img

    def do_export(self, images, filename):
        if self.export == True:
            frame, draw_frame, heatmap, thresh_heatmap, labels, ret_img = images
            fig = plt.figure(figsize=(15, 10))
            plt.subplot(2,3,1)
            plt.imshow(frame)
            plt.subplot(2,3,2)
            plt.imshow(draw_frame)
            plt.subplot(2,3,3)
            plt.imshow(heatmap, cmap='hot')
            plt.subplot(2,3,4)
            plt.imshow(thresh_heatmap, cmap='hot')
            plt.subplot(2,3,5)
            plt.imshow(labels, cmap='gray')
            plt.subplot(2,3,6)
            plt.imshow(ret_img)
            fig.tight_layout()
            plt.savefig(os.path.join('./output_images/', filename), format='png')

    def filter(self, frame, draw_frame, windows, reset):
        if reset == True:
            self.history.clear()

        self.history.append(windows)

        heatmap = np.zeros_like(frame[:, :, 0], dtype=np.float)
        for bboxes in self.history:
            heatmap = self.add_heat(heatmap, bboxes)

        if self.export == True:
            mpimg.imsave(os.path.join('./output_images', 'raw_heatmap.png'), heatmap, format='png', cmap='hot')

        thresh_heatmap = self.apply_threshold(heatmap)

        if self.export == True:
            mpimg.imsave(os.path.join('./output_images', 'thresh_heatmap.png'), thresh_heatmap, format='png', cmap='hot')

        labels = label(thresh_heatmap)

        if self.export == True:
            mpimg.imsave(os.path.join('./output_images', 'labels.png'), labels[0], format='png', cmap='gray')

        ret_img = self.draw_labeled_bboxes(draw_frame, labels)
        if self.export == True:
            mpimg.imsave(os.path.join('./output_images', 'labeled_bboxes_on.png'), ret_img, format='png')

        imgs_to_export = (frame, draw_frame, heatmap, thresh_heatmap, labels[0], ret_img)
        self.do_export(imgs_to_export, 'filter_combined.png')

        return self.draw_labeled_bboxes(draw_frame, labels)
