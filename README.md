# Keypoints-Detection
Detect keypoints on images using Mask RCNN neural network.

This research is aim for egocentric vision real-time fingertip detection using RGB image and without depth information.

The repository includes:

* Mask R-CNN using MobileNetV2 as backbone
* Redesign the model architecture by decrease the layer of convoulution
* Inspect the feature-map of each layer and implement it on `inspect_model.py` 
* Release a trained model for testing

## Performace
Mask R-CNN using MobilNetV2 as backbone trained on 640x480 input size

* 100 Proposals: 

|                       Model                       | Total parameters | Keypoints mask loss |  FPS  |
|:-------------------------------------------------:|:----------------:|:-------------------:|:-----:|
|           Resnet50 + 5 FPN + 8 Keypoints          |    59,754,910    |        3.708        | 22.63 |
| MobileNet with 17 conv layer + 5 FPN + 8 Keypoint |    37,158,046    |        3.902        | 29.41 |
| MobileNet with 12 conv layer + 5 FPN + 3 Keypoint |    24,820,134    |        3.633        | 32.95 |
|  MobileNet with 9 conv layer + 4 FPN + 3 Keypoint |    23,184,294    |        3.554        | 34.48 |
