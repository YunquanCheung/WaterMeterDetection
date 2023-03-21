import os
import shutil
import time
from pathlib import Path

import cv2
import torch
from numpy import random
from utils.general import (check_img_size, non_max_suppression, scale_coords, xyxy2xywh, polygon_non_max_suppression, polygon_scale_coords)
#from utils.general import (check_img_size, non_max_suppression, scale_coords, xyxy2xywh, plot_one_box)
from utils.torch_utils import time_synchronized
from utils.plots import polygon_plot_one_box
from utils.datasets import LoadStreams, LoadImages
from utils.torch_utils import select_device
from backend.flask_id2name import id2name

def predict(opt, model, img):
    out, source, view_img, save_img, save_txt, imgsz = \
        opt['output'], opt['source'], opt['view_img'], opt['save_img'], opt['save_txt'], opt['imgsz']


# Initialize
device = select_device(opt['device']) 
if os.path.exists(out):
    shutil.rmtree(out)  # delete output folder
os.makedirs(out)  # make new output folder
half = device.type != 'cpu'  # half precision only supported on CUDA

im0_shape = img.shape 

# Load model
# model = attempt_load(weights, map_location=device)  # load FP32 model
imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
if half:
    model.half()  # to FP16

# Set Dataloader
dataset = LoadImages(opt['source'], img_size=imgsz)

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

# Run inference
t0 = time.time()
# img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
# _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
for path, img, im0s, _ in dataset:
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    
    pred = model(img, augment=opt['augment'])[0] 

    pred = polygon_non_max_suppression(pred, opt['conf_thres'], opt['iou_thres'], classes=opt['classes'], agnostic=opt['agnostic_nms'])
   
    t2 = time_synchronized()

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        p, s, im0 = path, '', im0s
        save_path = str(Path(out) / Path(p).name) 
        txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0, 1, 0, 1, 0]]  
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :8] = polygon_scale_coords(img.shape[2:], det[:, :8], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, names[int(c)])  # add to string

            # Write results
            boxes_detected = []        
            for *xyxyxyxy, conf, cls in reversed(det):
                xyxyxyxy_list = (torch.tensor(xyxyxyxy).view(1, 8)).view(-1).tolist()  
                print(xyxyxyxy)
                boxes_detected.append({"name": id2name[int(cls.item())],
                                "conf": str(conf.item()),
                                "bbox": [int(xyxyxyxy_list[0]), int(xyxyxyxy_list[1]), int(xyxyxyxy_list[2]), int(xyxyxyxy_list[3]), int(xyxyxyxy_list[4]), int(xyxyxyxy_list[5]), int(xyxyxyxy_list[6]), int(xyxyxyxy_list[7])]
                                
                                })
                if save_txt:  # Write to file
                    xyxyxyxyn = (torch.tensor(xyxyxyxy).view(1, 8) / gn).view(-1).tolist()  
                    line = (cls, *xyxyxyxyn, conf) if conf else (cls, *xyxyxyxyn)   
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')   # label format

                if save_img or view_img:  # Add bbox to image
                
                    label = '%s %.2f' % (names[int(cls)], conf)
                    
                    polygon_plot_one_box(torch.tensor(xyxyxyxy).cpu().numpy(), im0, label=label, color=colors[0], line_thickness=3)
        # Print time (inference + NMS)
        print(xyxyxyxy)
        print(xyxyxyxy_list)
        print(torch.tensor(xyxyxyxy).cpu().numpy())
        print('%sDone. (%.3fs)' % (s, t2 - t1))
        # Save results (image with detections)
        if save_img:
            if dataset.mode == 'images':
                cv2.imwrite(save_path, im0)

results = {"results": boxes_detected}
print(results)
return results
