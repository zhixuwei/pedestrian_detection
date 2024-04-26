import os.path

from .yolox_base import YOLOXBase


class YOLOXNano(YOLOXBase):
    def __init__(self):
        super().__init__()
        self.depth = 0.33
        self.width = 0.25
        self.depth_wise = True
        self.model_path = "data/model/yolox_nano.pth"
        self.model = self.get_model()


class YOLOXTiny(YOLOXBase):
    def __init__(self):
        super().__init__()
        self.depth = 0.33
        self.width = 0.375
        self.model_path = "data/model/yolox_tiny.pth"
        self.model = self.get_model()
