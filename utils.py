"""
Mask R-CNN
Common utility functions and classes.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import numpy as np
import tensorflow as tf
import scipy.misc
import skimage.color
import skimage.io
import urllib.request
import shutil
import datetime
import cv2
from threading import Thread

# URL from which to download the latest COCO trained weights
COCO_MODEL_URL = "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"


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


############################################################
#  Human Pose
############################################################

def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:

        center = factor - 1
    else:
        center = factor - 0.5
        og = np.ogrid[:size, :size]
        return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)


def bilinear_upsample_weights(factor, number_of_classes):
    filter_size = factor * 2 - factor % 2
    weights = np.zeros((filter_size, filter_size, number_of_classes, number_of_classes),
                       dtype=np.float32)
    upsample_kernel = upsample_filt(filter_size)
    for i in range(number_of_classes):
        weights[:, :, i, i] = upsample_kernel
    return weights


def keypoint_to_mask(keypoints, height, width):
    """Convert keypoints to masks and it's weight.
       keypoints: [num_person, num_keypoint, 3].
       height,width: the generated mask shape

       Returns:
           keypoint_mask: A bool array of shape [height, width, num_person, num_keypoint] with
            one mask per joint..
           keypoint_weight: A int array of shape [num_person, num_keypoint] one value per joint
           0: not visible and without annotations
           1: not visible but with annotations
           2: visible and with annotations
       """
    shape = np.shape(keypoints)

    keypoint_mask = np.zeros([height, width, shape[0], shape[1]], dtype=bool)
    keypoint_weight = np.zeros([shape[0], shape[1]], dtype=int)
    for i in range(shape[0]):
        for j in range(shape[1]):
            J = keypoints[i, j]
            # print(J)
            if(J[2]):
                keypoint_mask[J[1], J[0], i, j] = 1
            keypoint_weight[i, j] = J[2]
    # keypoint_mask = np.reshape(keypoint_mask,[height,width,-1])
    # keypoint_weight = np.reshape(keypoint_weight,[-1])
    return keypoint_mask, keypoint_weight


############################################################
#  Bounding Boxes
############################################################

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        x1 = 0 if x1 < 0 else x1
        y1 = 0 if y1 < 0 else y1
        y2 = mask.shape[0] - 1 if y2 >= mask.shape[0] else y2
        x2 = mask.shape[1] - 1 if x2 >= mask.shape[1] else x2
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


def extract_fi_bboxes(mask, new_size, old_size, scale, padding):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    # mask=np.sum(mask,3)
    boxes = np.zeros([mask.shape[0], 4], dtype=np.int32)
    for i in range(mask.shape[0]):
        m = mask[i, :, :]
        # # Bounding box.
        # horizontal_indicies = np.where(np.any(m, axis=1))[0]
        # vertical_indicies = np.where(np.any(m, axis=0))[0]
        # if horizontal_indicies.shape[0]:
        #     x1, x2 = horizontal_indicies[[0, -1]]
        #     y1, y2 = vertical_indicies[[0, -1]]
        #     # x2 and y2 should not be part of the box. Increment by 1.
        #     x2 += 1
        #     y2 += 1
        # else:
        #     # No mask for this instance. Might happen due to
        #     # resizing or cropping. Set bbox to zeros
        #     x1, x2, y1, y2 = 0, 0, 0, 0
        # boxes[i] = np.array([y1, x1, y2, x2])
        vertical_indicies = m[np.where(m[:, 0] > 0), 0]
        x1 = vertical_indicies.min()
        x2 = vertical_indicies.max() + 1
        x1 -= 12
        x2 += 12
        if(x1 < 0):
            x1 = 0
        if (x2 > old_size[1]):
            x2 = old_size[1]
        horizontal_indicies = m[np.where(m[:, 1] > 0), 1]
        y1 = horizontal_indicies.min()
        y2 = horizontal_indicies.max() + 1
        y1 -= 12
        y2 += 12
        if(y1 < 0):
            y1 = 0
        if (y2 > old_size[0]):
            y2 = old_size[0]
        x1, y1 = resize_box(x1, y1, new_size, scale, padding)
        x2, y2 = resize_box(x2, y2, new_size, scale, padding)
        boxes[i] = np.array([y1, x1, y2, x2])
        # print(boxes[i])
    return boxes.astype(np.int32)


def resize_box(x, y, new_size, scale, padding):
    x = int(x * scale + 0.5)
    y = int(y * scale + 0.5)
    if (x >= new_size[1]):
        x = new_size[1] - 1
    if (y >= new_size[0]):
        y = new_size[0] - 1
    # padding
    x = x + padding[1][0]
    y = y + padding[0][0]
    return x, y


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    :return overlaps [boxes1.shape[0], boxes2.shape[0]]

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


def compute_overlaps_masks(masks1, masks2):
    '''Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    '''
    # flatten masks
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps


def non_max_suppression(boxes, scores, threshold):
    """Performs non-maximum supression and returns indicies of kept boxes.
    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != 'f':
        boxes = boxes.astype(np.float32)

    # Compute box areas
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)

    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indicies into ixs[1:], so add 1 to get
        # indicies into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indicies of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)]. Note that (y2, x2) is outside the box.
    deltas: [N, (dy, dx, log(dh), log(dw))]
    """
    boxes = boxes.astype(np.float32)
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= np.exp(deltas[:, 2])
    width *= np.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    return np.stack([y1, x1, y2, x2], axis=1)


def box_refinement_graph(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]
    """
    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = tf.log(gt_height / height)
    dw = tf.log(gt_width / width)

    result = tf.stack([dy, dx, dh, dw], axis=1)
    return result


def box_refinement(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]. (y2, x2) is
    assumed to be outside the box.
    """
    box = box.astype(np.float32)
    gt_box = gt_box.astype(np.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = np.log(gt_height / height)
    dw = np.log(gt_width / width)

    return np.stack([dy, dx, dh, dw], axis=1)


############################################################
#  Dataset
############################################################

class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:

    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...

    See COCODataset and ShapesDataset as examples.
    """

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{'source': '', 'id': 0, 'name': 'BG'}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert '.' not in source, 'Source name cannot contain a dot'
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info['id'] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            'source': source,
            'id': class_id,
            'name': class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            'id': image_id,
            'source': source,
            'path': path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.

        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """
        return ''

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ','.join(name.split(',')[:1])

        # self.class_ids [0-num_id-1]
        # self.source_class_ids{"coco":[0-num_id-1]}
        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c['name']) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        self.class_from_source_map = {'{}.{}'.format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    def append_data(self, class_info, image_info):
        self.external_to_class_id = {}
        for i, c in enumerate(self.class_info):
            for ds, id in c['map']:
                self.external_to_class_id[ds + str(id)] = i

        # Map external image IDs to internal ones.
        self.external_to_image_id = {}
        for i, info in enumerate(self.image_info):
            self.external_to_image_id[info['ds'] + str(info['id'])] = i

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's availble online for easy
        debugging.
        """
        return self.image_info[image_id]['path']

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        return image

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].

        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids

    def load_keypoints(self, image_id):
        """Load keypoints for the given image.

        Different datasets use different ways to store masks. Override this
        method to load keypoints and return them in the form of am
        array of coordinate(x,y) of shape [num_keypoints, 3].

        Returns:
            keypoints: A  array of coordinate and visibility [num_keypoints, 3] with
                (x,y, v) per instance.
            class_ids: a 1D array of class IDs of the person, always equal to [1].
        """
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        keypoints = np.empty([0, 0])
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return keypoints, mask, class_ids


def resize_image(image, min_dim=None, max_dim=None, padding=False):
    """
    Resizes an image keeping the aspect ratio.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    padding: If true, pads image with zeros so it's size is max_dim x max_dim

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    # Does it exceed max dim?
    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
    # Resize image and mask
    if scale != 1:
        image = scipy.misc.imresize(
            image, (round(h * scale), round(w * scale)))
    # Need padding?
    if padding:
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image, window, scale, padding


def resize_mask(mask, scale, padding):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    h, w = mask.shape[:2]
    mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask


def get_keypoints(img_category):
    """Get the COCO keypoints and their left/right flip coorespondence map."""
    # Keypoints are not available in the COCO json for the test split, so we
    # provide them here.
    fi_class_names_ = ['fingertip', 'joint']
    finger_index = [0, 1]

    all_index = {'finger': finger_index}

    index = all_index[img_category]
    keypoints = []
    for i in index:
        keypoints.append(fi_class_names_[i])

    finger_flip_map = {}

    keypoints_flip_map = {'finger': finger_flip_map}

    keypoint_flip_map = keypoints_flip_map[img_category]
    return keypoints, keypoint_flip_map


def flip_keypoints(keypoints, keypoint_flip_map, keypoint_coords, width):
    """Left/right flip keypoint_coords. keypoints and keypoint_flip_map are
    accessible from get_keypoints().
    keypoint_coords:[ni,_person, num_keypoint, 3]
    width: image_width
    """

    flipped_kps = keypoint_coords.copy()
    for lkp, rkp in keypoint_flip_map.items():
        lid = keypoints.index(lkp)
        rid = keypoints.index(rkp)
        flipped_kps[:, lid, :] = keypoint_coords[:, rid, :]
        flipped_kps[:, rid, :] = keypoint_coords[:, lid, :]

    # Flip x coordinates
    flipped_kps[:, :, 0] = width - flipped_kps[:, :, 0] - 1
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = np.where(flipped_kps[:, :, 2] == 0)
    flipped_kps[inds[0], inds[1], 0] = 0
    return flipped_kps


def resize_keypoints(keypoint, new_size, scale, padding):
    """Resizes a keypoint using the given scale and padding.
        Typically, you get the scale and padding from resize_image() to
        ensure both, the image and the mask, are resized consistently.
        keypoint: [num_person, num_keypoint, 3]
        scale: mask scaling factor
        padding: Padding to add to the mask in the form
                [(top, bottom), (left, right), (0, 0)]
        """
    keypoint_shape = np.shape(keypoint)
    num_person = keypoint_shape[0]
    num_keypoint = keypoint_shape[1]
    for i in range(num_person):
        for j in range(num_keypoint):
            x = keypoint[i, j, 0]
            y = keypoint[i, j, 1]
            vis = keypoint[i, j, 2]
            # scale
            x = int(x * scale + 0.5)
            y = int(y * scale + 0.5)
            if(x >= new_size[1]):
                x = new_size[1] - 1
            if(y >= new_size[0]):
                y = new_size[0] - 1
            # padding
            x = x + padding[1][0]
            y = y + padding[0][0]
            keypoint[i, j, :2] = [x, y]

    # keypoint[:,:,0] = np.array(keypoint[:,:,0]*scale + 0.5).astype(int)
    # keypoint[:,:,1] = np.array(keypoint[:,:,1]*scale + 0.5).astype(int)
    # X = keypoint[:,:,0]
    # Y = keypoint[:,:,1]
    # X[X>=new_size[1]] = new_size[1] -1
    # Y[Y>=new_size[0]] = new_size[0] - 1
    # X = X + padding[1,0]
    # Y = Y + padding[0,0]
    # keypoint[:, :, 0] =X
    # keypoint[:, :, 1] = Y

    return keypoint


def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to cut memory load.
    Mini-masks can then resized back to image scale using expand_masks()

    See inspect_data.ipynb notebook for more details.
    """
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception('Invalid bounding box with area of zero')
        m = scipy.misc.imresize(m.astype(float), mini_shape, interp='bilinear')
        # _positon = np.argmax(m)  # get the index of max in the a
        # m_index, n_index = divmod(_positon, mini_shape[0])
        # print("Max in oringal:", (m_index, n_index), m[m_index, n_index])
        mini_mask[:, :, i] = np.where(m >= 128, 1, 0)
    return mini_mask


def minimize_keypoint_mask(bbox, keypointmask, mini_shape):
    """Resize keypoint_mask to a smaller version to cut memory load.
        Mini-masks can then resized back to image scale using expand_masks()

        See inspect_data.ipynb notebook for more details.
        """
    mini_mask = np.zeros(mini_shape + (keypointmask.shape[2], keypointmask.shape[3],), dtype=bool)
    for i in range(keypointmask.shape[2]):
        for j in range(keypointmask.shape[3]):
            m = keypointmask[:, :, i, j]
            y1, x1, y2, x2 = bbox[i][:4]
            m = m[y1:y2, x1:x2]
            if m.size == 0:
                raise Exception('Invalid bounding box with area of zero')
            if m.sum() == 0:
                mini_mask[0, 0, i, j] = 1
                # mini_mask = mini_mask
            else:
                scale = np.asarray(mini_shape).astype(float) / m.shape
                cordys, cordxs = np.where(m == np.max(m))
                scale = np.asarray(mini_shape).astype(float) / m.shape
                cordys = (cordys * scale[0] + 0.5).astype(int)
                cordxs = (cordxs * scale[1] + 0.5).astype(int)
                cordys[cordys >= mini_shape[0]] = mini_shape[0] - 1
                cordxs[cordxs >= mini_shape[1]] = mini_shape[1] - 1
                final_y = np.mean(cordys).astype(int)
                final_x = np.mean(cordxs).astype(int)
                mini_mask[final_y, final_x, i, j] = 1
                # scale = np.asarray(mini_shape) / m.shape
                # cord = np.where(m == int(m.max()))
                # new_cord = np.array([cord[0] * scale[0], cord[1] * scale[1]], dtype=np.int32).reshape(2, )
                # mini_mask[new_cord[0], new_cord[1], i,j] = 1
    return mini_mask


def expand_keypoint_mask(bbox, mini_mask, image_shape):
    """Resizes mini keypoint masks back to image size. Reverses the change
        of minimize_mask().

        See inspect_data.ipynb notebook for more details.
    """
    keypoint_mask = np.zeros(image_shape[:2] + (mini_mask.shape[2], mini_mask.shape[3]))

    for i in range(keypoint_mask.shape[2]):
        for j in range(keypoint_mask.shape[3]):
            m = mini_mask[:, :, i, j]
            y1, x1, y2, x2 = bbox[i][:4]

            h = y2 - y1
            w = x2 - x1
            result = np.sum(m)
            if(result):
                cordys, cordxs = np.where(m == np.max(m))
                scale = np.asarray([h, w]).astype(float) / m.shape
                cordys = (cordys * scale[0] + 0.5).astype(int)
                cordxs = (cordxs * scale[1] + 0.5).astype(int)

                cordys[cordys >= h] = h - 1
                cordxs[cordxs >= w] = w - 1
                m = np.zeros(np.asarray([h, w]).astype(int), dtype=bool)
                # print("m shape:", np.shape(m))
                final_y = np.mean(cordys).astype(int)
                final_x = np.mean(cordxs).astype(int)
                m[final_y, final_x] = 1
            else:
                m = np.zeros([h, w])

            keypoint_mask[y1:y2, x1:x2, i, j] = m

    return keypoint_mask


def expand_mask(bbox, mini_mask, image_shape):
    """Resizes mini masks back to image size. Reverses the change
    of minimize_mask().

    See inspect_data.ipynb notebook for more details.
    """
    mask = np.zeros(image_shape[:2] + (mini_mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mini_mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        h = y2 - y1
        w = x2 - x1
        m = scipy.misc.imresize(m.astype(float), (h, w), interp='bilinear')
        # _positon = np.argmax(m)  # get the index of max in the a
        # m_index, n_index = divmod(_positon, w)
        # print("Max in resize:", (m_index, n_index), m[m_index, n_index])
        mask[y1:y2, x1:x2, i] = np.where(m >= 128, 1, 0)

    return mask


# TODO: Build and use this function to reduce code duplication
def mold_mask(mask, config):
    pass


def unmold_mask(mask, bbox, image_shape):
    """Converts a mask generated by the neural network into a format similar
    to it's original shape.
    mask: [height, width, channel] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.

    Returns a binary mask with the same size as the original image.
    """
    threshold = 0.5
    y1, x1, y2, x2 = bbox
    mask = scipy.misc.imresize(
        mask, (y2 - y1, x2 - x1), interp='bilinear').astype(np.float32) / 255.0
    mask = np.where(mask >= threshold, 1, 0).astype(np.uint8)

    # Put the mask in the right location.
    full_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = mask
    return full_mask


def unmold_keypoint_mask(keypoints_prob, bbox, image_shape, keypoint_mask_shape=(56, 56), keypoint_threshold=0.08):
    """Converts a mask generated by the neural network into a format similar
    to it's original shape.
    keypoints_probe: [num_keypoints, 56*56] of type float.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.
    image_shape:
    mask: [height, width, channel] of type float. A small, typically 28x28 mask.
    keypoint_mask_shape:
    keypoint_threshold: the threshold for filter the low confident keypoint
    Returns
    keypoints: [num_keypoints, 3] for (x , y, valid)
    """

    keypoints_label = np.argmax(keypoints_prob, 1)
    keypoint_score = np.max(keypoints_prob, 1)

    J_y = keypoints_label // keypoint_mask_shape[1]
    J_x = keypoints_label % keypoint_mask_shape[1]
    box_height = float(bbox[2] - bbox[0])
    box_width = float(bbox[3] - bbox[1])
    x_scale = box_width / keypoint_mask_shape[1]
    y_scale = box_height / keypoint_mask_shape[0]
    x_shift = bbox[1]
    y_shift = bbox[0]
    J_x = np.array(x_scale * J_x + 0.5).astype(int) + x_shift
    J_y = np.array(y_scale * J_y + 0.5).astype(int) + y_shift
    J_v = np.array(keypoint_score > keypoint_threshold).astype(int)
    keypoints = np.stack([J_x, J_y, J_v], axis=1)

    return keypoints


############################################################
#  Anchors
############################################################

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)


############################################################
#  Miscellaneous
############################################################

def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.

    x: [rows, columns].
    """
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]


def compute_ap(gt_boxes, gt_class_ids, gt_masks,
               pred_boxes, pred_class_ids, pred_scores, pred_masks,
               iou_threshold=0.5):
    """Compute Average Precision at a set IoU threshold (default 0.5).

    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding and sort predictions by score from high to low
    # TODO: cleaner to do zero unpadding upstream
    gt_boxes = trim_zeros(gt_boxes)
    gt_masks = gt_masks[..., :gt_boxes.shape[0]]
    pred_boxes = trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[..., indices]

    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = compute_overlaps_masks(pred_masks, gt_masks)

    # Loop through ground truth boxes and find matching predictions
    match_count = 0
    pred_match = np.zeros([pred_boxes.shape[0]])
    gt_match = np.zeros([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] == 1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = 1
                pred_match[i] = 1
                break

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])

    return mAP, precisions, recalls, overlaps


def compute_recall(pred_boxes, gt_boxes, iou):
    """Compute the recall at the given IoU threshold. It's an indication
    of how many GT boxes were found by the given prediction boxes.

    pred_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    gt_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    """
    # Measure overlaps
    overlaps = compute_overlaps(pred_boxes, gt_boxes)
    iou_max = np.max(overlaps, axis=1)
    iou_argmax = np.argmax(overlaps, axis=1)
    positive_ids = np.where(iou_max >= iou)[0]
    matched_gt_boxes = iou_argmax[positive_ids]

    recall = len(set(matched_gt_boxes)) / gt_boxes.shape[0]
    return recall, positive_ids


# ## Batch Slicing
# Some custom layers support a batch size of 1 only, and require a lot of work
# to support batches greater than 1. This function slices an input tensor
# across the batch dimension and feeds batches of size 1. Effectively,
# an easy way to support batches > 1 quickly with little code modification.
# In the long run, it's more efficient to modify the code to support large
# batches and getting rid of this function. Consider this a temporary solution
def batch_slice(inputs, graph_fn, batch_size, names=None):
    """Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.

    inputs: list of tensors. All must have the same first dimension length
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result


def download_trained_weights(coco_model_path, verbose=1):
    """Download COCO trained weights from Releases.

    coco_model_path: local path of COCO trained weights
    """
    if verbose > 0:
        print('Downloading pretrained model to ' + coco_model_path + ' ...')
    with urllib.request.urlopen(COCO_MODEL_URL) as resp, open(coco_model_path, 'wb') as out:
        shutil.copyfileobj(resp, out)
    if verbose > 0:
        print('... done downloading pretrained model!')
