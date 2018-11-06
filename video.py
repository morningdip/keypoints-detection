import os
import time
import numpy as np
import pandas as pd
import cv2

import utils
import model as modellib
import visualize
import progressbar
import terminal_color as tc
from config import Config
from model import log
from PIL import Image
from keras import backend as K
from progressbar import AnimatedMarker, Bar, ETA, Percentage, SimpleProgress


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Root directory of the project
ROOT_DIR = '../'

fi_class_names = ['finger']
class_names = ['fingertip', 'joint1', 'joint2']
index = [0, 1, 2]

widgets = [Percentage(), ' (', SimpleProgress(format='%(value)02d/%(max_value)d'), ') ', AnimatedMarker(markers='◢◣◤◥'), ' ', Bar(marker='>'), ' ', ETA()]

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs/{}_logs".format(fi_class_names[0]))
model_path = os.path.join(
    ROOT_DIR, "model/mask_rcnn_{}.h5".format(fi_class_names[0]))
results_path = os.path.join(ROOT_DIR, 'results')


class FingerConfig(Config):
    # Give the configuration a recognizable name
    NAME = "finger"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_KEYPOINTS = len(index)
    KEYPOINT_MASK_SHAPE = [56, 56]
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 24 key_point

    RPN_TRAIN_ANCHORS_PER_IMAGE = 150
    VALIDATION_STPES = 100
    STEPS_PER_EPOCH = 100
    MINI_MASK_SHAPE = (56, 56)
    KEYPOINT_MASK_POOL_SIZE = 7

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    MASK_SHAPE = [28, 28]
    WEIGHT_LOSS = True
    KEYPOINT_THRESHOLD = 0.005
    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 128
    DETECTION_MAX_INSTANCES = 1


if __name__ == '__main__':

    # config of model
    inference_config = FingerConfig()
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(
        mode='inference', config=inference_config, model_dir=MODEL_DIR)
    # Get path to saved weights
    # Load trained weights (fill in path to trained weights here)
    assert model_path != '', 'Provide path to trained weights'
    print('Loading weights from ', model_path)
    model.load_weights(model_path, by_name=True)

    null_image = np.zeros((1, 1, 3))
    results = model.detect_keypoint([null_image], verbose=0)

    stream = cv2.VideoCapture('http://140.115.54.125:5566/videostream.cgi?.mjpg')
    while True:
        # Capture frame-by-frame
        grabbed, frame = stream.read()
        if not grabbed:
            break

        results = model.detect_keypoint([frame], verbose=0)
        r = results[0]

        save_img_path = os.path.join(results_path, 'result.png')

        keypoints_image = visualize.get_keypoints_image(frame, r['rois'], r['keypoints'], r['class_ids'], class_names)
        visualize.save_keypoints(
            frame, save_img_path,
            r['rois'], r['keypoints'], r['class_ids'],
            class_names)
        cv2.imshow('', keypoints_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

stream.release()
cv2.destroyAllWindows()
