import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    if color_space != 'RGB':
        cv2_space = eval('cv2.COLOR_RGB2' + color_space)
        feature_image = cv2.cvtColor(img, cv2_space)
    else:
        feature_image = np.copy(img)
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel()
    # Return the feature vector
    return features

image = mpimg.imread('../test_images/cutout4.jpg')
feature_vec = bin_spatial(image, color_space='RGB', size=(32, 32))

# Plot features
plt.plot(feature_vec)
plt.title('Spatially Binned Features')
# plt.show()
