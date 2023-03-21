# OritentedWaterMeterReadingRecognition

This repository is based on https://github.com/ultralytics/yolov5, with modification to enable oriented prediction boxes.

- Modified origin YOLOv5 for oriented object detection
- Created a custom Water Meter Reading Recognition dataset by collecting and labeling water meter pictures from real scenes.
- Trained and tested the oriented YOLOv5 model
- Deployed the oriented YOLOv5 model to build a Flask web application
- Added water meter reading recognition to include the corresponding reading


## New Web Demo

<img src="./newDemo.gif" alt="OrientedWaterMeter" style="zoom:150%;" />

## Environment

To use this project, you first need to install the requirements and CUDA Extension:

```
# clone project
git clone https://github.com/YunquanCheung/WaterMeterDetection.git

# install requirements
pip install -r requirements.txt

# install CUDA extensions
cd utils/iou_cuda
python setup.py install
```

## Usage

#### Predict

```bash
!python polygon_detect.py --weights polygon-rtg.pt --img 1024 --conf 0.75 \
    --source data/images/WM --iou-thres 0.4 --hide-labels
```

#### Train

```bash
!python polygon_train.py --weights polygon-rtg.pt --cfg polygon_yolov5s_wm.yaml \
    --data polygon_wm.yaml --hyp hyp.wm.yaml --img-size 1024 \
    --epochs 3 --batch-size 12 --noautoanchor --polygon --cache
```

-- cfg    : models

-- data:  training location

--hyp   : hyperparameter selection

#### Flask Web

```bash
! sh run.sh
```

## Acknowledgments

This project is based on the ultralytics/YOLOv5


