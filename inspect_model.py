import os
import time
import numpy as np
import pandas as pd
import utils
import model as modellib
import visualize
import progressbar
import matplotlib
import matplotlib.pyplot as plt

from config import Config
from PIL import Image
from keras import backend as K


import terminal_color as tc


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Root directory of the project
ROOT_DIR = '../'

fi_class_names = ['finger']
class_names = ['fingertip', 'joint']
index = [0, 1]

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, 'logs/{}_logs'.format(fi_class_names[0]))
model_path = os.path.join(ROOT_DIR, 'model/mobilev2_mask_rcnn_finger_0600_v3.h5')
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

    BACKBONE = 'mobilenetv2'

    IMAGE_MIN_DIM = 480
    IMAGE_MAX_DIM = 640

    RPN_ANCHOR_SCALES = (20, 40, 80, 160, 320)

    # RPN_TRAIN_ANCHORS_PER_IMAGE = 150
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64
    VALIDATION_STPES = 100
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

    # Last Backbone Layer
    if inference_config.BACKBONE in ["resnet50", "resnet101"]:
        laBaLa = "res4w_out"
    if inference_config.BACKBONE in ["mobilenetv1"]:
        laBaLa = "conv_pw_1_bn"
    if inference_config.BACKBONE in ["mobilenetv2"]:
        laBaLA = "conv_pw_1_bn"

    for index, x in enumerate(range(0, 2)):
        image = dataset_test.load_image(x)
        image = np.expand_dims(image, axis=0)
        #image = np.array(image).reshape(1, 640, 640, 3)

        # Get activations of a few sample layers
        activations = model.run_graph([image], [
            ("input_image", model.keras_model.get_layer("input_image").output),
            ("conv3", model.keras_model.get_layer("conv3").output)],
            TEST_MODE='inference')

    # Backbone feature map
    '''
    NUM_LAYERS = 12
    for i in range(NUM_LAYERS):
        layer = "conv_pw_{}_bn".format(i+1)
        BB_activations = model.run_graph([image], [(layer, model.keras_model.get_layer(layer).output)], TEST_MODE='inference')
        print(BB_activations[layer].shape)
        visualize.display_images(np.transpose(BB_activations[layer][0,:,:,:16], [2, 0, 1])*10000)
    '''


    layer = "res11"
    activations = model.run_graph([image], [(layer, model.keras_model.get_layer(layer).output)], TEST_MODE='inference')
    print(activations[layer].shape)

    visualize.display_images(np.transpose(activations[layer][0, :, :, :12], [2, 0, 1]))

    K.clear_session()
