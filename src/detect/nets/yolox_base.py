import torch.nn as nn

from .yolox.models import YOLOX, YOLOPAFPN, YOLOXHead


def init_yolo(M):
    for m in M.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03


class YOLOXBase:
    def __init__(self):
        self.num_classes = 80
        self.act = "silu"
        self.depth = 1
        self.width = 1
        self.depth_wise = False
        self.model = None

    def get_model(self):
        in_channels = [256, 512, 1024]
        # NANO model use depthwise = True, which is main difference.
        backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act, depthwise=self.depth_wise, )
        head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act, depthwise=self.depth_wise)
        self.model = YOLOX(backbone, head)
        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
