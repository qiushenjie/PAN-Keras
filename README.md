# Keras implementation of the paper "Efficient and Accurate Arbitrary-Shaped Text Detection with Pixel Aggregation Network"
## Requirements
* Keras==2.1.5
* Tensorflow==1.8.0
* opencv==3.4.2.16

## Data Preparation
train.txt
```
1.jpg xmin1 ymin1 xmax1 ymin1 xmax1 ymax1 xmin1 ymax1 xmin2 ymin2 xmax2 ymin2 xmax2 ymax2 xmin2 ymax2......
2.jpg xmin1 ymin1 xmax1 ymin1 xmax1 ymax1 xmin1 ymax1 xmin2 ymin2 xmax2 ymin2 xmax2 ymax2 xmin2 ymax2......
...
```

val.txt
```
3.jpg xmin1 ymin1 xmax1 ymin1 xmax1 ymax1 xmin1 ymax1 xmin2 ymin2 xmax2 ymin2 xmax2 ymax2 xmin2 ymax2......
4.jpg xmin1 ymin1 xmax1 ymin1 xmax1 ymax1 xmin1 ymax1 xmin2 ymin2 xmax2 ymin2 xmax2 ymax2 xmin2 ymax2......
...
```
(and you can convert your .xml into .txt by ./utils/process_xml.py)
## Train
```sh
python train_multithreading.py
```

## Test and Prediction
use [H5toPB.py](H5toPB.py.py) to convert .h5 into .pb, which will make reference more comfortable(mabye not).

1.[val_eval.py](valeval.py) is used to predict your val dataset and calculate the accuracy and so on.

2.[predict_pb.py](predict_pb.py) is used to predict your val dataset and show the results in real time.

if you want to predict a single image, please find the fuction in [predict_pb.py](predict_pb.py)


### reference
1. https://github.com/WenmuZhou/PAN.pytorch
