import torch.nn as nn


class SE(nn.Module):
    def __init__(self, channels, reduction=2):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x)
        # print(y.shape)
        y = self.excitation(y)
        # print(y.shape)
        return x * y
