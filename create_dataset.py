import os
import csv
from natsort import natsorted

json_folfer = '../data/val/images'
output_folder = '../data/val/annotations/test.csv'

ext_name = '.jpg'
test_files = [img for img in os.listdir(json_folfer) if img.endswith(ext_name)]

test_files = natsorted(test_files)
print(test_files)

with open(output_folder, 'w', newline='') as csvfile:
    fieldnames = ['image_id', 'image_category']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for test in test_files:
        writer.writerow({'image_id': 'images/' + test,
                         'image_category': 'finger'})
