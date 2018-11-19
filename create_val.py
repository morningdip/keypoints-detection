import os
import csv
import re
import progressbar as pb

from natsort import natsorted

widgets = [pb.Percentage(), ' (', pb.SimpleProgress(format='%(value)02d/%(max_value)d'), ') ', pb.AnimatedMarker(markers='◢◣◤◥'), ' ', pb.Bar(marker='>'), ' ', pb.ETA()]


train_folder = '../data/val/label'
output_folder = '../data/val/annotations/val.csv'
ext_name = '.txt'

IMG_WIDTH = 640
IMG_HEIGHT = 480

txt_files = [txt for txt in os.listdir(train_folder) if txt.endswith(ext_name)]
txt_files = natsorted(txt_files)

with open(output_folder, 'w', newline='') as csvfile:
    fieldnames = ['image_id', 'image_category', 'fingertip', 'joint']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    pbar = pb.ProgressBar(widgets=widgets, max_value=len(txt_files), redirect_stdout=True)
    pbar.start()

    for idx, label_file in enumerate(txt_files):
        with open(os.path.join(train_folder, label_file)) as f:
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