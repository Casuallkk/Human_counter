"""观察网络结构"""
import torch
from torchsummary import summary
from nets.yolo import YoloBody

if __name__ == "__main__":
    input_shape = [640, 640]
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    num_classes = 80

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = YoloBody(anchors_mask, num_classes, False, 1).to(device)
    summary(m, (3, 640, 640))
