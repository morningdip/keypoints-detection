import os
import numpy as np
import pandas as pd
import tensorflow as tf
import model as modellib
import progressbar
from config import Config
import utils
from PIL import Image
from keras.backend.tensorflow_backend import set_session

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

gpu_options = tf.GPUOptions(allow_growth=True)
set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Root directory of the project
ROOT_DIR = '../'

class_names = ['finger']

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, 'logs/{}_logs'.format(class_names[0]))
SELF_MODEL_PATH = os.path.join(ROOT_DIR, 'model/mobilev2_mask_rcnn_finger_0600_v5.h5')
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'model/mask_rcnn_coco.h5')

# Numbers of keypoints
keypoints_class_names = ['fingertip', 'joint']
index = [0, 1]


class FingerConfig(Config):
    IMAGE_CATEGORY = class_names[0]
    # Give the configuration a recognizable name
    NAME = 'finger'

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1

    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 4

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1000

    # Number of validation steps to run at the end of every training epoch.
    VALIDATION_STEPS = 100

    # Backbone Architecture,
    # Currently supported: ['resnet50','resnet101', 'mobilenetv1','mobilenetv2']
    BACKBONE = 'mobilenetv2'

    # The strides of each layer of the FPN Pyramid.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + keypoint
    NUM_KEYPOINTS = len(index)

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (20, 40, 80, 160, 320)

    # Ratios of anchors at each cell (width/height)
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 200
    POST_NMS_ROIS_INFERENCE = 100

    MINI_MASK_SHAPE = (56, 56)

    KEYPOINT_MASK_SHAPE = [56, 56]
    KEYPOINT_MASK_POOL_SIZE = 7
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    MASK_SHAPE = [28, 28]

    IMAGE_MIN_DIM = 480
    IMAGE_MAX_DIM = 640

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    # TRAIN_ROIS_PER_IMAGE = 20

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "keypoint_mrcnn_mask_loss": 1.
    }

    WEIGHT_LOSS = True
    KEYPOINT_THRESHOLD = 0.005


def pic_height_width(filepath):
    fp = open(filepath, 'rb')
    im = Image.open(fp)
    fp.close()
    x, y = im.size
    if im.mode == 'RGB':
        return x, y
    else:
        return False, False


class FingerDataset(utils.Dataset):

    def load_finger(self, category):
        # Add classes
        for i, class_name in enumerate(class_names):
            self.add_class('finger', i + 1, class_name)

        if category == 'train':
            data_path = '../data/train/'
            annotations = pd.read_csv('../data/train/annotations/train.csv')
            '''
            annotations = annotations.append(
                pd.read_csv('../data/train/annotations/data_scaling.csv'), ignore_index=True)
            annotations = annotations.append(
                pd.read_csv('../data/train/annotations/data_flip.csv'), ignore_index=True)
            '''
        elif category == 'val':
            data_path = '../data/val/'
            annotations = pd.read_csv('../data/val/annotations/val.csv')
        else:
            pass

        annotations = annotations.loc[annotations['image_category'] == class_names[0]]
        annotations = annotations.reset_index(drop=True)

        np.random.seed(42)
        indices = np.random.permutation(annotations.shape[0])
        annotations = annotations.iloc[indices]
        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().

        annotations = annotations.reset_index(drop=True)

        widgets = [progressbar.Percentage(), ' (', progressbar.SimpleProgress(format='%(value)02d/%(max_value)d'), ') ',
                   progressbar.AnimatedMarker(markers='.oO@* '),
                   ' ', progressbar.Bar(marker='>'), ' ', progressbar.ETA()]
        pbar = progressbar.ProgressBar(widgets=widgets, max_value=annotations.shape[0], redirect_stdout=True)
        pbar.start()

        for x in range(annotations.shape[0]):
            # bg_color, shapes = self.random_image(height, width)
            id = annotations.loc[x, 'image_id']
            category = annotations.loc[x, 'image_category']
            pbar.update(x)
            # print('loading image:%d/%d' % (x, annotations.shape[0]))
            im_path = os.path.join(data_path, id)

            # height, width = cv2.imread(im_path).shape[0:2]
            width, height = pic_height_width(im_path)

            key_points = []
            for key_point in annotations.loc[x, keypoints_class_names].values:
                loc_cat = [int(j) for j in key_point.split('_')]
                key_points.append(loc_cat)

            self.add_image('finger', image_id=id, path=im_path,
                           width=width, height=height,
                           key_points=key_points, image_category=category)
        pbar.finish()

    def load_image(self, image_id):
        info = self.image_info[image_id]
        image = Image.open(info['path'])
        image = np.array(image)
        return image

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info['source'] == 'finger':
            return info['key_points'], info['image_category']
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_keypoints(self, image_id):
        info = self.image_info[image_id]
        # image_category = info['image_category']
        key_points = np.array(info['key_points'])

        keypoints = []
        keypoint = []
        class_ids = []

        for part_num, bp in enumerate(key_points):
            if(bp[2] == -1):
                keypoint += [0, 0, 0]
            else:
                keypoint += [bp[0], bp[1], bp[2]]
        keypoint = np.reshape(keypoint, (-1, 3))
        keypoints.append(keypoint)
        class_ids.append(1)

        if class_ids:
            keypoints = np.array(keypoints, dtype=np.int32)
            class_ids = np.array(class_ids, dtype=np.int32)
            return keypoints, 0, class_ids
        else:
            return super(self.__class__).load_keypoints(image_id)


if __name__ == '__main__':
    config = FingerConfig()
    # Training dataset
    dataset_train = FingerDataset()
    dataset_train.load_finger(category='train')
    dataset_train.prepare()

    # Validation dataset
    dataset_val = FingerDataset()
    dataset_val.load_finger(category='val')
    dataset_val.prepare()

    print('Classes: {}.'.format(dataset_train.class_names))
    print('Train Images: {}.'.format(len(dataset_train.image_ids)))
    print('Valid Images: {}.'.format(len(dataset_val.image_ids)))

    model = modellib.MaskRCNN(
        mode='training', config=config, model_dir=MODEL_DIR)

    # Which weights to start with?
    init_with = 'self_mdoel'    # imagenet, coco, or last
    if init_with == 'coco':
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(
            COCO_MODEL_PATH, by_name=False, exclude=['mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])
    elif init_with == 'last':
        # Load the last model you trained and continue training
        model.load_weights(model.find_last()[1], by_name=True)
    elif init_with == 'self_mdoel':
        model.load_weights(SELF_MODEL_PATH, by_name=True)

    print('Train all')
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=100,
                layers='all')
    print('Train all')
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=200,
                layers='all')
    print('Train all')
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 100,
                epochs=300,
                layers='heads')

    '''
    # Training - WarmUp Stage
    print("> Warm Up all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=100,
                layers='all')

    # Training - Stage 1
    print("> Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=200,
                layers='heads')

    # Training - Stage 2
    # Finetune layers  stage 4 and up
    print("> Fine tune {} stage 4 and up".format(config.BACKBONE))
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=300,
                layers="4+")

    # Training - Stage 3
    # Fine tune all layers
    print("> Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=400,
                layers='all')

    # Training - Stage 3
    # Fine tune all layers
    print("> Fine tune all layers till the break of Dawn")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 100,
                epochs=500,
                layers='all')
    '''
