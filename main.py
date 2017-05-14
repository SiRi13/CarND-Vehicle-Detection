import os
import cv2
import glob
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from sklearn.externals import joblib
from moviepy.video.io.VideoFileClip import VideoFileClip

from classes.trainer import ClassifierTrainer
from classes.detector import VehicleDetector
from classes.filter import VehicleFilter
import classes.utilities as utils

if not (os.path.exists('./classes/settings/parameters.p') and os.path.exists('./classes/settings/svc.p')):
    print('no parameters! start training')
    trainer = ClassifierTrainer()
    trainer.train()
    print('export settings')
    trainer.export_settings()
    print('done')

# %% load settings
print('load settings')
parameters = joblib.load('./classes/settings/parameters.p')
svc = joblib.load('./classes/settings/svc.p')
print('done')

# %% export images for writeup
detector = VehicleDetector(svc['classifier'], svc['scaler'], parameters, VehicleFilter(thresh=1, hist_len=1, export=True))
img = mpimg.imread('./test_images/test6.jpg')
draw_img = np.copy(img)

# %% process frame
draw_img = detector.process_frame(draw_img)
plt.figure(figsize=(15,10))
plt.imshow(draw_img)
plt.show()

# %% draw sliding windows
windows = detector.get_sliding_windows(draw_img, separately=True)
windows0 = utils.draw_boxes(np.copy(img), windows[0], color=(255, 0, 0))
plt.imsave('./output_images/test4_windows0.png', windows0, format='png')

# %% test images
detector = VehicleDetector(svc['classifier'], svc['scaler'], parameters, VehicleFilter(thresh=1.0, hist_len=1))
print('detect on test images')
test_images = glob.glob('./test_images/test*.jpg')
for image in test_images:
    img = mpimg.imread(image)
    output_image = detector.process_frame(img, reset=True)
    out_name = os.path.join('./output_images', os.path.basename(image))
    mpimg.imsave(out_name.replace('jpg', 'png'), output_image, format='png')
print('done')

# %% test video
detector = VehicleDetector(svc['classifier'], svc['scaler'], parameters, VehicleFilter(thresh=5.0, hist_len=10))
video = 'test_video.mp4'
video_output = './output_videos/' + video
clip = VideoFileClip(video)
output_clip = clip.fl_image(detector.process_frame)
output_clip.write_videofile(video_output, audio=False)
print('done')

# %% project video
detector = VehicleDetector(svc['classifier'], svc['scaler'], parameters, VehicleFilter(thresh=10.0, hist_len=10))
video = 'project_video.mp4'
video_output = './output_videos/' + video
clip = VideoFileClip(video) #.subclip(39, 40)
output_clip = clip.fl_image(detector.process_frame)
output_clip.write_videofile(video_output, audio=False)
print('done')
