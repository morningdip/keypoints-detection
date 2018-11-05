import cv2
import pandas as pd
import os
import numpy as np

data_path = '../data/train/'
save_name = 'images_flip'
annotations_save_path = os.path.join(data_path, 'annotations')
save_path = os.path.join(data_path, save_name)

fi_class_names_ = ['fingertip', 'joint1', 'joint2']
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


for x in range(annotations.shape[0]):
    id = annotations.loc[x, 'image_id']
    category = annotations.loc[x, 'image_category']
    print('loading image:%d/%d' % (x, annotations.shape[0]))
    im_path = os.path.join(data_path, id)

    key_points = []
    for key_point in annotations.loc[x, fi_class_names_].values:
        loc_cat = [int(j) for j in key_point.split('_')]
        key_points.append(loc_cat)

    img = cv2.imread(im_path)
    y_mid = img.shape[0] / 2
    # Flipped horizontally
    img = cv2.flip(img, 0)

    for i in range(len(key_points)):
        if(key_points[i][2] != -1):
            key_points[i][1] = y_mid + (y_mid - key_points[i][1])
            key_points[i][1] = int(key_points[i][1])
            # cv2.rectangle(img=img, pt1=(key_points[i][0], key_points[i][1]), pt2=(key_points[i][0], key_points[i][1]), thickness=2, color=(255, 0, 0))
    key_points = keypoint_to_str(key_points)
    new_dir = id.replace('images', save_name)
    relust_info = [new_dir, category] + key_points
    csv_all.append(relust_info)
    cv2.imwrite(os.path.join(data_path, new_dir), img)

columns = ['image_id', 'image_category']
columns.extend(fi_class_names_)
point_to_csv = pd.DataFrame(data=np.array(csv_all).reshape([-1, 5]), columns=columns)
point_to_csv.to_csv(annotations_save_path + '/data_flip.csv', index=False)
