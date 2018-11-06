import cv2
import pandas as pd
import os
import numpy as np
import progressbar

from progressbar import AnimatedMarker, Bar, ETA, Percentage, SimpleProgress

widgets = [Percentage(), ' (', SimpleProgress(format='%(value)02d/%(max_value)d'), ') ', AnimatedMarker(markers='◢◣◤◥'), ' ', Bar(marker='>'), ' ', ETA()]

data_path = '../data/train/'
save_name = 'images_scaling'
annotations_save_path = os.path.join(data_path, 'annotations')
save_path = os.path.join(data_path, save_name)

fi_class_names_ = ['fingertip', 'joint']
csv_all = []

annotations = pd.read_csv('../data/train/annotations/train.csv')

if (os.path.exists(annotations_save_path)) is False:
    os.makedirs(annotations_save_path)

if (os.path.exists(save_path)) is False:
    os.makedirs(save_path)


def keypoint_to_str(keypoint):
    list_keypoint = []
    for x in keypoint:
        list_keypoint.append(str(x[0]) + '_' + str(x[1]) + '_' + str(x[2]))
    return list_keypoint


for x in progressbar.progressbar(range(0, annotations.shape[0]), widgets=widgets, redirect_stdout=True):
    id = annotations.loc[x, 'image_id']
    category = annotations.loc[x, 'image_category']
    im_path = os.path.join(data_path, id)

    key_points = []
    for key_point in annotations.loc[x, fi_class_names_].values:
        loc_cat = [int(j) for j in key_point.split('_')]
        key_points.append(loc_cat)

    img = cv2.imread(im_path)

    re_size = np.random.uniform(0.5, 0.9)
    img = cv2.resize(img, (0, 0), fx=re_size, fy=re_size)

    for i in range(len(key_points)):
        if(key_points[i][2] != -1):
            key_points[i][0] *= re_size
            key_points[i][1] *= re_size
            key_points[i][0] = int(key_points[i][0])
            key_points[i][1] = int(key_points[i][1])
            # cv2.rectangle(img=img, pt1=(key_points[i][0], key_points[i][1]), pt2=(key_points[i][0], key_points[i][1]), thickness=10, color=(255, 0, 0))
    key_points = keypoint_to_str(key_points)
    new_dir = id.replace("images", save_name)
    relust_info = [new_dir, category] + key_points
    csv_all.append(relust_info)
    cv2.imwrite(os.path.join(data_path, new_dir), img)


columns = ['image_id', 'image_category']
columns.extend(fi_class_names_)
point_to_csv = pd.DataFrame(data=np.array(csv_all).reshape([-1, 4]), columns=columns)
point_to_csv.to_csv(annotations_save_path + '/data_scaling.csv', index=False)
