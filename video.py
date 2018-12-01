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
from keras import backend as K
from progressbar import AnimatedMarker, Bar, ETA, Percentage, SimpleProgress

from queue import Queue
from threading import Thread
import datetime


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Root directory of the project
ROOT_DIR = '../'

fi_class_names = ['finger']
class_names = ['fingertip', 'joint']
index = [0, 1]

widgets = [Percentage(), ' (', SimpleProgress(format='%(value)02d/%(max_value)d'), ') ', AnimatedMarker(markers='◢◣◤◥'), ' ', Bar(marker='>'), ' ', ETA()]

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs/{}_logs".format(fi_class_names[0]))
model_path = os.path.join(ROOT_DIR, "model/mask_rcnn_{}_0600.h5".format(fi_class_names[0]))
#model_path = os.path.join(ROOT_DIR, "model/mobile_mask_rcnn_finger_0228.h5")
results_path = os.path.join(ROOT_DIR, 'results')


class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()


class WebcamVideoStream:
    def __init__(self, src, width, height):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


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

    BACKBONE = 'resnet50'

    IMAGE_MIN_DIM = 480
    IMAGE_MAX_DIM = 640

    RPN_ANCHOR_SCALES = (20, 40, 80, 160, 320)

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


def worker(input_q, output_q):
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
    model.detect_keypoint([null_image], verbose=0)

    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_q.put((frame_rgb, model.detect_keypoint([frame_rgb], verbose=0)))

    fps.stop()
    K.clear_session()


if __name__ == '__main__':

    input_q = Queue(5)  # fps is better if queue is higher but then more lags
    output_q = Queue()
    for i in range(1):
        t = Thread(target=worker, args=(input_q, output_q))
        t.daemon = True
        t.start()

    video_capture = WebcamVideoStream(src='http://140.115.54.125:5566/videostream.cgi?.mjpg',
                                      width=640, height=480).start()

    frame_num = 0
    fps = FPS().start()

    while True:
        frame = video_capture.read()

        t = time.time()

        if frame_num % 4 == 0:
            input_q.put(frame)
        else:
            pass

        if output_q.empty():
            print('pass')
            pass  # fill up queue
        else:
            print('processing')
            tmp, data = output_q.get()
            # print(data[0])
            r = data[0]

            keypoints_image = visualize.get_keypoints_image(tmp, r['rois'], r['keypoints'], r['class_ids'], class_names)
            cv2.imshow('Demo', keypoints_image)

        frame_num += 1

        fps.update()

        print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

video_capture.stop()
cv2.destroyAllWindows()
