from __future__ import print_function

import cv2
import imutils
import numpy as np
from imutils.object_detection import non_max_suppression
from .detect import Detector


class PeopleDetect(object):

    def __init__(self):
        self.model = Detector()

    def detect(self, image):
        detect_result = self.model.detect_image(image)
        orig, num = self.model.draw_image(image.copy(), detect_result)

        return orig, image, num

    def detectImg(self, imagePath):
        image = cv2.imread(imagePath)
        return self.detect(image)

    def detectVideo(self, frame):
        return self.detect(frame)
