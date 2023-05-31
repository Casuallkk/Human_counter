"""对代码进行了优化"""
import torch
import torch.nn as nn

from nets.net_processer import ConvBnSilu, MultiConcatBlock, TransitionBlock


class Backbone(nn.Module):
    def __init__(self, transition_channels, block_channels, pretrained=False):
        super().__init__()
        ids = [-1, -3, -5, -6]
        self.stem = nn.Sequential(
            ConvBnSilu(3, transition_channels, 3, 1),
            ConvBnSilu(transition_channels, transition_channels * 2, 3, 2),
            ConvBnSilu(transition_channels * 2, transition_channels * 2, 3, 1))
        self.dark2 = nn.Sequential(
            ConvBnSilu(transition_channels * 2, transition_channels * 4, 3, 2),
            MultiConcatBlock(transition_channels * 4, block_channels * 2, transition_channels * 8, ids=ids))
        self.dark3 = nn.Sequential(
            TransitionBlock(transition_channels * 8, transition_channels * 4),
            MultiConcatBlock(transition_channels * 8, block_channels * 4, transition_channels * 16, ids=ids))
        self.dark4 = nn.Sequential(
            TransitionBlock(transition_channels * 16, transition_channels * 8),
            MultiConcatBlock(transition_channels * 16, block_channels * 8, transition_channels * 32, ids=ids))
        self.dark5 = nn.Sequential(
            TransitionBlock(transition_channels * 32, transition_channels * 16),
            MultiConcatBlock(transition_channels * 32, block_channels * 8, transition_channels * 32, ids=ids))
        if pretrained:
            url = 'https://github.com/bubbliiiing/yolov7-pytorch/releases/download/v1.0/yolov7_backbone_weights.pth'
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
            self.load_state_dict(checkpoint, strict=False)
            print("Load weights from " + url.split('/')[-1])

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        # feat1：80, 80, 512
        feat1 = self.dark3(x)
        # feat2：40, 40, 1024
        feat2 = self.dark4(feat1)
        # feat3：20, 20, 1024
        feat3 = self.dark5(feat2)
        return feat1, feat2, feat3
