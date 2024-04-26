import cv2
import numpy as np
import torch

from .nets.yolox.utils import postprocess
from .nets.yolox_tools import YOLOXNano, YOLOXTiny


class Detector:
    def __init__(self, mode="fast"):
        self.net = YOLOXNano() if mode == "fast" else YOLOXTiny()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = self.net.model_path
        self.model = self.net.get_model()
        ckpt = torch.load(self.model_path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device)
        self.model.eval()

        self.test_size = (416, 416)
        self.num_classes = self.net.num_classes
        self.conf_thr = 0.3
        self.nms_thr = 0.3

    def letterbox_image(self, img):
        if len(img.shape) == 3:
            padded_img = np.ones((self.test_size[0], self.test_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(self.test_size, dtype=np.uint8) * 114

        r = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
        padded_img = padded_img.transpose((2, 0, 1))
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img

    def detect_image(self, img):
        img = self.letterbox_image(img)
        img = torch.from_numpy(img).unsqueeze(0).to(self.device).float()
        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(outputs, self.num_classes, self.conf_thr, self.nms_thr, class_agnostic=True)
        return outputs[0] if isinstance(outputs, list) else outputs

    def draw_image(self, img, outputs, cls_conf=0.55):
        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        target_num = 0
        if outputs is None:
            return img, target_num
        outputs = outputs.cpu().numpy()
        outputs = outputs[outputs[:, 6] == 0]
        outputs[:, 0:4] /= ratio

        target_num = 0
        for output in outputs:
            bbox = output[:4]
            cls = output[6]
            score = output[4] * output[5]
            if score < cls_conf:
                continue
            x0 = int(bbox[0])
            y0 = int(bbox[1])
            x1 = int(bbox[2])
            y1 = int(bbox[3])
            target_num += 1

            cv2.rectangle(img, (x0, y0), (x1, y1), (77, 171, 255), 2)
        return img, target_num
