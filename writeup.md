# **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[data_exploration]: ./output_images/car_not_car.png
[test1_hog]: ./output_images/test1_hog.png
[test1]: ./output_images/test1.png
[test2]: ./output_images/test2.png
[test6]: ./output_images/test6.png
[sliding_windows]: ./output_images/test4_windows0.png
[all_detections]: ./output_images/all_detected_windows.png
[windows_and_heat]: ./output_images/windows_and_heat.png
[labeled_windows]: ./output_images/labels.png
[filter_combined]: ./output_images/filter_combined.png

### [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

---

## Writeup
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for training is contained in file [trainer.py](classes/trainer.py). The *ClassifierTrainer* class loads _vehicle_ and _non-vehicle_ images on instantiation.

![alt text][data_exploration]

By calling the `train()` function, it starts to extract all features from previously loaded images. Then the feature vectors get scaled by the *StandardScaler()*. After shuffling the data it gets split into training and testing sets.

#### 2. Explain how you settled on your final choice of HOG parameters.

After trying different set ups, like the one from the lesson in color space _RGB_, 9 orientations, 8 pixels per cell, 2 cells per block and spatial binning as well as color histograms, I stayed with following color space and parameters:

- Color space: `YUV`
- HOG parameters of `orientations=11`, `pixels_per_cell=(16, 16)`, `cells_per_block=(2, 2)` and `hog_channel="ALL"`
- No spatial Binning
- No color histograms

![alt text][test1_hog]

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The classifier is also trained in function `train()` in file [trainer.py](classes/trainer.py).
After scaling them I used a *LinearSVC* to fit the training data and predict the labels.
The result was:  
```
Using: 11 orientations 16 pixels per cell and 2 cells per block
Feature vector length: 1188
2.11 Seconds to train SVC...
Test Accuracy of SVC =  0.9749
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

To create windows I used the function `slide_window()` in file [utilities.py](classes/utilities.py), which returns a list of windows to search in.
The `search_windows` function is located in class *VehicleDetector* in file [detector.py](classes/detector.py). It uses the windows from `slide_window()` to slide over the lower part of the image or frame and tries to detect any vehicles.

I first tried three different windows sizes for vehicles which are far away, not so far away and close to the camera. But it lead to a very slow processing of the frames since it had to try a lot of windows.
After some experimenting I chose only one window size of 76x76 with 70% overlap from 360 to 700 in y direction.

![alt text][sliding_windows]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on one scale using YUV 3-channel HOG features without spatially binned color or histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][test1]
![alt text][test2]
![alt text][test6]

To improve performance I tried to compute the HoG features only once per image but did not succeed.
Instead I reduced the sliding window sizes to one instead of three different sizes which improved speed a little.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to the project video](./output_videos/project_video.mp4) and the [test video](./output_images/test_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The class *VehicleFilter* in file [filter.py](classes/filter.py) contains measurements against false positives and to combine multiple detections to one bounding box.
I implemented a history which is added to the heat map. This stabilized the boxes and eliminated a lot of the false positives.
After thresholding the heat map I used `scipy.ndimage.measurements.label` to identify individual blobs.


Here's an example result showing the heat map from the test images, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

Here is one test image with its corresponding heat map and labels:

![alt text][filter_combined]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The major problem of my pipeline is performance. It maxes out at about three frames per second which will never work in a real time environment.
Another point to improvement is the region of interest which could be enhanced by using the lane finding algorithm.
