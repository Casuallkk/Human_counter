"""自己写的, 里面包括se, cbam, eca以及ca四种"""
import torch
from torch import nn
import math



class se_block(nn.Module):
    """
    Squeeze-and-Excitation Networks:
    paper: https://arxiv.org/pdf/1709.01507.pdf
    """
    def __init__(self, channel, ratio=16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, False),
            # //是向下取整， /会有小数
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, False),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        fc = self.avg_pool(x).view(b, c)
        fc = self.fc(fc).view(b, c, 1, 1)
        return x * fc


class ChannelAttention(nn.Module):
    """通道注意力"""
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, (1, 1), bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, (1, 1), bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力"""
    def __init__(self, kernel=7):
        super(SpatialAttention, self).__init__()
        assert kernel in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel == 7 else 1
        self.conv = nn.Conv2d(2, 1, (kernel, kernel), padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 最大池化
        x = torch.cat([avg_out, max_out], dim=1)
        # 堆叠: (2, 1, 26, 26) & (2, 1, 26, 26)->(2, 2, 26, 26)
        conv = self.conv(x)
        return self.sigmoid(conv)


class cbam_block(nn.Module):
    """
    CBAM: Convolutional Block Attention Module
    paper: https://arxiv.org/pdf/1807.06521.pdf
    """
    def __init__(self, channel, ratio=16, kernel=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio)
        self.spatialattention = SpatialAttention(kernel)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x



class eca_block(nn.Module):
    """
    ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
    paper:https://arxiv.org/abs/1910.03151
    codes:https://github.com/BangguWu/ECANet
    """
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel = int(abs((math.log(channel, 2) + b) / gamma))
        kernel = kernel if kernel % 2 else kernel + 1
        # 自适应计算kernel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel,
                              padding=(kernel - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class ca_block(nn.Module):
    """
    Coordinate Attention for Efficient Mobile Network Design
    paper: https://arxiv.org/pdf/2103.02907.pdf
    codes: https://github.com/houqb/CoordAttention
    """
    def __init__(self, channel, reduction=16):
        super(ca_block, self).__init__()
        self.conv = nn.Conv2d(channel, channel // reduction, (1, 1), 1,
                                  bias=False)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(channel // reduction)
        self.F_h = nn.Conv2d(channel // reduction, channel, (1, 1), 1,
                             bias=False)
        self.F_w = nn.Conv2d(channel // reduction, channel, (1, 1), 1,
                             bias=False)
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()
        # 对h和w方向上做平均池化，其中permute(0, 1, 3, 2)是将第二维和第4维换位置
        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)
        # 对h和w方向上对堆叠，再标准化，最后用Relu激活
        x_cat_conv_relu = self.relu(self.batch_norm(self.conv(torch.cat((x_h, x_w), 3))))
        # 将h与w方向上的特征分开
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)
        # 将之前换用permute的换回来，再做一次标准化
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
        # 将h和w方向上的特征扩展
        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out



if __name__ == '__main__':
    model = ca_block(512)
    print(model)
    inputs = torch.ones(2, 512, 26, 26)
    outputs = model(inputs)
    print(outputs)