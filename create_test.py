import os
import csv
from natsort import natsorted

json_folfer = '../data/test2/images'
output_folder = '../data/test2/annotations/test2.csv'

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
