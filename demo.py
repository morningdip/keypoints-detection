import os
import time
import numpy as np
import cv2
import model as modellib
import visualize

from config import Config
from keras import backend as K
from queue import Queue
from threading import Thread
from utils import FPS, WebcamVideoStream
from PIL import Image, ImageDraw, ImageFont


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Root directory of the project
ROOT_DIR = '../'

fi_class_names = ['finger']
class_names = ['fingertip', 'joint']
index = [0, 1]

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs/{}_logs".format(fi_class_names[0]))
model_path = os.path.join(ROOT_DIR, 'model/mobilev2_mask_rcnn_finger_0900_v5.h5')


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
    NUM_CLASSES = 1 + 1

    BACKBONE = 'mobilenetv2'

    IMAGE_MIN_DIM = 480
    IMAGE_MAX_DIM = 640

    RPN_ANCHOR_SCALES = (20, 40, 80, 160, 320)

    RPN_TRAIN_ANCHORS_PER_IMAGE = 64
    VALIDATION_STEPS = 100
    STEPS_PER_EPOCH = 100
    MINI_MASK_SHAPE = (56, 56)
    KEYPOINT_MASK_POOL_SIZE = 7

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 200
    POST_NMS_ROIS_INFERENCE = 100

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

    #fps = FPS().start()
    while True:
        #fps.update()
        frame = input_q.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_q.put((frame_rgb, model.detect_keypoint([frame_rgb], verbose=0)))

    #fps.stop()
    K.clear_session()


if __name__ == '__main__':

    font = ImageFont.truetype('../font/candy-gothic.ttf', 50)

    cv2.namedWindow('demo', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('demo', 640, 480)

    input_q = Queue(2)  # fps is better if queue is higher but then more lags
    output_q = Queue()

    for i in range(1):
        t = Thread(target=worker, args=(input_q, output_q))
        t.daemon = True
        t.start()

    video_src = 'http://140.115.54.125:8090/videostream.cgi?.mjpg'
    video_capture = WebcamVideoStream(src=video_src, width=640, height=480).start()

    frame_num = 0
    fps = FPS().start()

    while True:
        frame = video_capture.read()
        predicted_result = ''
        input_q.put(frame)

        t = time.time()

        if output_q.empty():
            pass
        else:
            img, data = output_q.get()

            r = data[0]

            if frame_num % 2 == 0:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img, text = visualize.get_keypoint_text(img, r['rois'], r['keypoints'], r['scores'])

                if text:
                    predicted_result = text

                img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)

                msg = predicted_result
                w, h = draw.textsize(msg, font=font)

                draw.rectangle(((0, 0), (640, h + 5)), fill=(53, 58, 67))
                draw.text(((640 - w) / 2, 0), msg, font=font, fill=(227, 229, 233))
                img = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)

                cv2.imshow('demo', img)
            else:
                pass

        frame_num += 1
        fps.update()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    video_capture.stop()
    cv2.destroyAllWindows()
