import os
import glob
import matplotlib.image as mpimg

from sklearn.externals import joblib
from moviepy.video.io.VideoFileClip import VideoFileClip

from classes.trainer import ClassifierTrainer
from classes.detector import VehicleDetector
from classes.filter import VehicleFilter

if not (os.path.exists('./classes/settings/parameters.p') and os.path.exists('./classes/settings/svc.p')):
    print('no parameters! start training')
    trainer = ClassifierTrainer()
    trainer.train()
    print('export settings')
    trainer.export_settings()
    print('done')

print('load settings')
parameters = joblib.load('./classes/settings/parameters.p')
svc = joblib.load('./classes/settings/svc.p')
print('done')


# %% test images
detector = VehicleDetector(svc['classifier'], svc['scaler'], parameters, VehicleFilter(thresh=5.0, hist_len=1))
print('detect on test images')
test_images = glob.glob('./test_images/test*.jpg')
for image in test_images:
    img = mpimg.imread(image)
    output_image = detector.process_frame(img)
    out_name = os.path.join('./output_images', os.path.basename(image))
    mpimg.imsave(out_name.replace('jpg', 'png'), output_image, format='png')
print('done')

# %% test video
detector = VehicleDetector(svc['classifier'], svc['scaler'], parameters, VehicleFilter(thresh=10.0, hist_len=10))
video = 'test_video.mp4'
video_output = './output_videos/' + video
clip = VideoFileClip(video)
output_clip = clip.fl_image(detector.process_frame)
output_clip.write_videofile(video_output, audio=False)
print('done')
