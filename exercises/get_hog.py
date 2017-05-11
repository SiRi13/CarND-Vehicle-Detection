import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import utils
from skimage.feature import hog

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        # Use skimage.hog() to get both features and a visualization
        features, hog_image = hog(img, orientations=orient,
                                    pixels_per_cell=(pix_per_cell, pix_per_cell),
                                    cells_per_block=(cell_per_block, cell_per_block),
                                    visualise=True, feature_vector=feature_vec)
        return features, hog_image
    else:
        # Use skimage.hog() to get features only
        features, _ = hog(img, orientations=orient,
                                    pixels_per_cell=(pix_per_cell, pix_per_cell),
                                    cells_per_block=(cell_per_block, cell_per_block),
                                    visualise=False, feature_vector=feature_vec)
        return features

cars, notcars = utils.get_images()

# %% Generate a random index to look at a car image
ind = np.random.randint(0, len(cars))
# Read in the image
image = mpimg.imread(cars[ind])
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# %% Define HOG parameters
orient = 9
pix_per_cell = 8
cell_per_block = 4
# Call our function with vis=True to see an image output
features, hog_image = get_hog_features(gray, orient, pix_per_cell, cell_per_block,
                                        vis=True, feature_vec=True)

# %% Plot the examples
fig = plt.figure()
plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('Example Car Image')
plt.subplot(122)
plt.imshow(hog_image, cmap='gray')
plt.title('HOG Visualization')
plt.show()
