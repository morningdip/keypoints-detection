import os
import csv
import re
import argparse

from natsort import natsorted
from progressbar import AnimatedMarker, Bar, ETA, Percentage, SimpleProgress, ProgressBar

widgets = [Percentage(), ' (', SimpleProgress(format='%(value)02d/%(max_value)d'), ') ', AnimatedMarker(markers='.oO@* '), ' ', Bar(marker='>'), ' ', ETA()]


def parse_arguments():
    parser = argparse.ArgumentParser(description='Create the annotations file for dataset.')
    parser.add_argument(
        '--type',
        '-t',
        type=str,
        nargs='?',
        help='Type of dataset, should be \'train\', \'test\', or \'val\'',
        default='test',
    )

    parser.add_argument(
        '--name',
        '-n',
        type=str,
        nargs='?',
        help='Name of dataset, should be \'train\', \'test\', or \'val\'',
        default='test',
    )

    return parser.parse_args()


def run():
    args = parse_arguments()

    type_dataset = args.type
    name = args.name
    assert type_dataset in ['train', 'test', 'val'], 'Argument Type not exist.'

    if type_dataset == 'test':
        test_dataset(name)
    elif type_dataset == 'train':
        trainval_dataset(type_dataset)
    elif type_dataset == 'val':
        trainval_dataset(type_dataset)


def test_dataset(name):
    img_dir = '../data/{}/images'.format(name)
    out_dir = '../data/{}/annotations/test.csv'.format(name)
    assert os.path.isdir(img_dir), 'Dataset directory is not exist.'

    test_files = [img for img in os.listdir(img_dir)]

    test_files = natsorted(test_files)
    print(test_files)

    with open(out_dir, 'w', newline='') as csvfile:
        field_names = ['image_id', 'image_category']
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()

        for test in test_files:
            writer.writerow({'image_id': 'images/' + test,
                             'image_category': 'finger'})


def trainval_dataset(type_dataset):
    label_dir = '../data/{}/label'.format(type_dataset)
    out_dir = '../data/{}/annotations/train.csv'.format(type_dataset)
    ext_name = '.txt'

    IMG_WIDTH = 640
    IMG_HEIGHT = 480

    txt_files = [txt for txt in os.listdir(label_dir) if txt.endswith(ext_name)]
    txt_files = natsorted(txt_files)

    with open(out_dir, 'w', newline='') as csvfile:
        field_names = ['image_id', 'image_category', 'fingertip', 'joint']
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()

        pbar = ProgressBar(widgets=widgets, max_value=len(txt_files), redirect_stdout=True)
        pbar.start()

        for idx, label_file in enumerate(txt_files):
            with open(os.path.join(label_dir, label_file)) as f:
                content = f.readlines()

            content = [x.strip() for x in content]

            for idx2, line in enumerate(content):
                try:
                    img_name, _, _, _, _, fingertip_x, fingertip_y, joint_x, joint_y, _, _, _, _, _, _ = map(
                        lambda x: x if not re.match('^\d+?\.\d+?$', x) else float(x), line.split())
                except ValueError:
                    print('Error! Line: {} in File: {}'.format(idx2, label_file))
                    break

                fingertip = '_'.join(map(str, (int(fingertip_x * IMG_WIDTH), int(fingertip_y * IMG_HEIGHT), 2)))
                joint = '_'.join(map(str, (int(joint_x * IMG_WIDTH), int(joint_y * IMG_HEIGHT), 2)))

                writer.writerow({'image_id': 'images/' + img_name,
                                 'image_category': 'finger',
                                 'fingertip': fingertip,
                                 'joint': joint})
            pbar.update(idx)
        pbar.finish()


if __name__ == "__main__":
    run()
