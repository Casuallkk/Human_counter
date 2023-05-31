## 室内人数计数

基于Yolo v7 PyTorch实现。

### 性能情况

| 训练数据集          | 权值文件名称                                                                                                            | 测试数据集        | 输入图片大小  | mAP 0.5:0.95 | mAP 0.5 |
|:--------------:|:-----------------------------------------------------------------------------------------------------------------:|:------------:|:-------:|:------------:|:-------:|
| COCO-Train2017 | [yolov7_weights.pth](https://github.com/bubbliiiing/yolov7-pytorch/releases/download/v1.0/yolov7_weights.pth)     | COCO-Val2017 | 640x640 | 50.7         | 69.2    |
| COCO-Train2017 | [yolov7_x_weights.pth](https://github.com/bubbliiiing/yolov7-pytorch/releases/download/v1.0/yolov7_x_weights.pth) | COCO-Val2017 | 640x640 | 52.4         | 70.5    |

### 所需环境

torch>=1.7.1

python=3.9

注：requirement.txt里面为我的环境里使用的各种依赖版本。torch=2.0.0+cu118。实测只要大于1.7.1即可。其他库其实装最新的就行，不会有冲突。

### 预测

运行prediect.py即可，默认模式为`predict`，即单张图片预测。默认测试图片为根目录下`img/street.jpg`。

<img title="" src="file:///C:/Users/雷馨月/AppData/Roaming/marktext/images/2023-05-31-15-04-19-image.png" alt="" data-align="inline" width="444">

注：如需使用自己训练的权重，在yolo.py文件里面，修改model_path和classes_path使其对应训练好的文件；**model_path对应权值文件，classes_path是model_path对应分的类**。



### 训练

##### 1. 数据集的准备及处理

**本代码使用VOC格式进行训练，需要VOC格式的数据集。** 运行voc_annotation.py即可获得train.txt和val.txt。默认训练集和测试集比例为9：1，默认训练集和验证集比例为9：1。

##### 2. 训练

运行train.py   



### 评估

运行get_map.py即可获得评估结果，评估结果会保存在map_out文件夹中

## Reference

https://github.com/WongKinYiu/yolov7

https://github.com/bubbliiiing/yolov7-pytorch
