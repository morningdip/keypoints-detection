import os
import time
import numpy as np
import pandas as pd

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


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Root directory of the project
ROOT_DIR = '../'

fi_class_names = ['finger']
class_names = ['fingertip', 'joint']
index = [0, 1]

widgets = [Percentage(), ' (', SimpleProgress(format='%(value)02d/%(max_value)d'), ') ', AnimatedMarker(markers='◢◣◤◥'), ' ', Bar(marker='>'), ' ', ETA()]

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, 'logs/{}_logs'.format(fi_class_names[0]))
model_path = os.path.join(ROOT_DIR, 'model/mask_rcnn_finger_0900.h5')
results_path = os.path.join(ROOT_DIR, 'results')


def pic_height_width(filepath):
    fp = open(filepath, 'rb')
    im = Image.open(fp)
    fp.close()
    x, y = im.size
    if im.mode == 'RGB':
        return x, y
    else:
        return False, False


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


class FingerTestDataset(utils.Dataset):

    def load_finger_test(self):
        test_data_path = '../data/test2'
        # Add classes
        for i, class_name in enumerate(fi_class_names):
            self.add_class('finger', i + 1, class_name)
        # annotations = pd.read_csv('../data/test2/annotations/test.csv')
        annotations = pd.read_csv(os.path.join(test_data_path, 'annotations', 'test.csv'))
        annotations = annotations.loc[annotations['image_category'] == fi_class_names[0]]
        annotations = annotations.reset_index(drop=True)

        for x in range(annotations.shape[0]):
            id = annotations.loc[x, 'image_id']
            category = annotations.loc[x, 'image_category']
            im_path = os.path.join(test_data_path, id)
            width, height = pic_height_width(im_path)
            self.add_image("finger", image_id=id, path=im_path,
                           width=width, height=height, image_category=category)

    def load_image(self, image_id):
        info = self.image_info[image_id]
        image = Image.open(info['path'])
        image = np.array(image)
        return image


if __name__ == '__main__':
    tcolor = tc.TerminalColor()
    dataset_test = FingerTestDataset()
    dataset_test.load_finger_test()
    dataset_test.prepare()

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

    pro_times = []

    for x in progressbar.progressbar(range(0, dataset_test.num_images), widgets=widgets, redirect_stdout=True):
        image = dataset_test.load_image(x)
        category = dataset_test.image_info[x]['image_category']
        image_id = dataset_test.image_info[x]['id']
        image_name, image_ext = os.path.splitext(
            os.path.basename(dataset_test.image_info[x]['id']))

        # print(, '- 0.13 seconds')

        null_image = np.zeros((1, 1, 3))
        results = model.detect_keypoint([null_image], verbose=0)

        start_time = time.time()
        results = model.detect_keypoint([image], verbose=0)
        end_time = time.time()

        pro_times.append(end_time - start_time)

        tcolor.printmc(('RED', 'GREEN'), 'Each image spend: ', '{}'.format(end_time - start_time))

        r = results[0]

        '''
        log("image", image)
        log("rois", r['rois'])
        log("keypoints", r['keypoints'])
        log("class_ids", r['class_ids'])
        '''

        save_img_path = os.path.join(results_path, image_name + '.png')

        visualize.save_fingertip_keypoint(
            image, save_img_path,
            r['rois'], r['keypoints'], r['class_ids'],
            dataset_test.class_names)

        pass

    meam_time = np.mean(pro_times)
    tcolor.printmc(('RED', 'Blue'), 'Mean FPS: ', '{}'.format(1 / meam_time))

    K.clear_session()
