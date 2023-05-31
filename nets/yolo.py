"""对代码进行了优化, 将ConvBnSilu，MultiConcatBlock，TransitionBlock，RepConv封装进net_processer.py"""
import torch
import torch.nn as nn

from nets.backbone import Backbone
from nets.net_processer import ConvBnSilu, MultiConcatBlock, TransitionBlock, RepConv, fuse_conv_and_bn
from nets.Attention import se_block, cbam_block, eca_block, ca_block

attention_blocks = [se_block, cbam_block, eca_block, ca_block]


class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = ConvBnSilu(c1, c_, 1, 1)
        self.cv2 = ConvBnSilu(c1, c_, 1, 1)
        self.cv3 = ConvBnSilu(c_, c_, 3, 1)
        self.cv4 = ConvBnSilu(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        # 采用不同大小的核进行最大池化，以扩大网络的感受野
        self.cv5 = ConvBnSilu(4 * c_, c_, 1, 1)
        self.cv6 = ConvBnSilu(c_, c_, 3, 1)
        # 输出通道数为c2
        self.cv7 = ConvBnSilu(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


class YoloBody(nn.Module):
    """搭建yolo网络"""
    def __init__(self, anchors_mask, num_classes, pretrained=False, index=0):
        super(YoloBody, self).__init__()
        transition_channels = 32
        block_channels = 32
        panet_channels = 32
        e = 2
        ids = [-1, -2, -3, -4, -5, -6]
        conv = RepConv

        self.backbone = Backbone(transition_channels, block_channels, pretrained=pretrained)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.sppcspc = SPPCSPC(transition_channels * 32, transition_channels * 16)
        self.conv_for_P5 = ConvBnSilu(transition_channels * 16, transition_channels * 8)
        self.conv_for_feat2 = ConvBnSilu(transition_channels * 32, transition_channels * 8)
        self.conv3_for_upsample1 = MultiConcatBlock(transition_channels * 16, panet_channels * 4,
                                                    transition_channels * 8, e=e, ids=ids)
        self.conv_for_P4 = ConvBnSilu(transition_channels * 8, transition_channels * 4)
        self.conv_for_feat1 = ConvBnSilu(transition_channels * 16, transition_channels * 4)
        self.conv3_for_upsample2 = MultiConcatBlock(transition_channels * 8, panet_channels * 2,
                                                    transition_channels * 4, e=e, ids=ids)
        self.down_sample1 = TransitionBlock(transition_channels * 4, transition_channels * 4)
        self.conv3_for_downsample1 = MultiConcatBlock(transition_channels * 16, panet_channels * 4,
                                                      transition_channels * 8, e=e, ids=ids)
        self.down_sample2 = TransitionBlock(transition_channels * 8, transition_channels * 8)
        self.conv3_for_downsample2 = MultiConcatBlock(transition_channels * 32, panet_channels * 8,
                                                      transition_channels * 16, e=e, ids=ids)
        self.rep_conv_1 = conv(transition_channels * 4, transition_channels * 8, 3, 1)
        self.rep_conv_2 = conv(transition_channels * 8, transition_channels * 16, 3, 1)
        self.rep_conv_3 = conv(transition_channels * 16, transition_channels * 32, 3, 1)
        self.yolo_head_P3 = nn.Conv2d(transition_channels * 8, len(anchors_mask[2]) * (5 + num_classes), (1, 1))
        self.yolo_head_P4 = nn.Conv2d(transition_channels * 16, len(anchors_mask[1]) * (5 + num_classes), (1, 1))
        self.yolo_head_P5 = nn.Conv2d(transition_channels * 32, len(anchors_mask[0]) * (5 + num_classes), (1, 1))
        self.index = index
        if 1 <= index <= 4:
            self.feat1_attention = attention_blocks[index - 1](512)
            self.feat2_attention = attention_blocks[index - 1](1024)
            self.feat3_attention = attention_blocks[index - 1](1024)
            self.upsample1_attention = attention_blocks[index - 1](256)
            self.upsample2_attention = attention_blocks[index - 1](256)

    def fuse(self):
        # print('Fusing layers... ')
        for m in self.modules():
            if isinstance(m, RepConv):
                m.fuse_repvgg_block()
            elif type(m) is ConvBnSilu and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.fuseforward
        return self

    def forward(self, x):
        feat1, feat2, feat3 = self.backbone.forward(x)
        if 1 <= self.index <= 4:
            feat1 = self.feat1_attention(feat1)
            feat2 = self.feat2_attention(feat2)
            feat3 = self.feat3_attention(feat3)
        # 加强特征提取网络
        P5 = self.sppcspc(feat3)
        P5_conv = self.conv_for_P5(P5)
        P5_upsample = self.upsample(P5_conv)

        if 1 <= self.index <= 4:
            P5_upsample = self.upsample1_attention(P5_upsample)

        P4 = torch.cat([self.conv_for_feat2(feat2), P5_upsample], 1)

        P4 = self.conv3_for_upsample1(P4)

        if 1 <= self.index <= 4:
            P4 = self.upsample2_attention(P4)

        P4_conv = self.conv_for_P4(P4)
        P4_upsample = self.upsample(P4_conv)

        P3 = torch.cat([self.conv_for_feat1(feat1), P4_upsample], 1)
        P3 = self.conv3_for_upsample2(P3)


        P3_downsample = self.down_sample1(P3)

        P4 = torch.cat([P3_downsample, P4], 1)

        P4 = self.conv3_for_downsample1(P4)


        P4_downsample = self.down_sample2(P4)

        P5 = torch.cat([P4_downsample, P5], 1)
        P5 = self.conv3_for_downsample2(P5)

        P3 = self.rep_conv_1(P3)
        P4 = self.rep_conv_2(P4)
        P5 = self.rep_conv_3(P5)
        # 第三个特征层
        # y3=(batch_size, 75, 80, 80)
        out2 = self.yolo_head_P3(P3)
        # 第二个特征层
        # y2=(batch_size, 75, 40, 40)
        out1 = self.yolo_head_P4(P4)
        # 第一个特征层
        # y1=(batch_size, 75, 20, 20)
        out0 = self.yolo_head_P5(P5)

        return [out0, out1, out2]
