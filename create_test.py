import os
import csv
from natsort import natsorted

json_folfer = '../data/ipcam/images'
output_folder = '../data/ipcam/annotations/test.csv'

test_files = [img for img in os.listdir(json_folfer)]

test_files = natsorted(test_files)
print(test_files)

with open(output_folder, 'w', newline='') as csvfile:
    fieldnames = ['image_id', 'image_category']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for test in test_files:
        writer.writerow({'image_id': 'images/' + test,
                         'image_category': 'finger'})
