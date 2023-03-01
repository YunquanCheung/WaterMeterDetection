from utils.autoanchor import polygon_kmean_anchors

nl = 3 # number of anchor layers
na = 3 # number of anchors
img_size = 640 # image size for training and testing

datacfg = "data/polygon_ucas.yaml" #data存放的位置
anchors = polygon_kmean_anchors(datacfg, n=nl*na, gen=3000, img_size=img_size)
print(anchors.reshape(nl, na*2).astype(int))
#请把这个anchors得到的值放在 models文件夹下面的 polygon_yolov5s_convert.yaml下
print('\nPlease Copy the anchors to your model configuration polygon_yolov5*.yaml')