import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
import cv2
import time

from src.detect.peopledetect import PeopleDetect
from PIL import Image, ImageDraw, ImageFont


class VideoReadThread(QThread):
    signalFrame = pyqtSignal(object, object, int)
    signalFailed = pyqtSignal(str)

    def __init__(self, parent=None):
        super(VideoReadThread, self).__init__(parent)
        self.work = False
        self.video_path = ""
        self.people_detect = PeopleDetect()

    def threadStart(self, path):
        self.video_path = path
        self.start()

    def threadStop(self):
        self.work = False

    def drawText(self, text, img):
        cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pilimg = Image.fromarray(cv2img)
        draw = ImageDraw.Draw(pilimg)
        font = ImageFont.truetype("data/font/SimHei.ttf", 20, encoding="utf-8")
        draw.text((100, 50), text, (255, 0, 0), font=font)
        cv2charimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
        return cv2charimg

    def run(self):
        try:
            self.work = True
            self.capture = cv2.VideoCapture(self.video_path)
            # self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            success, frame = self.capture.read()
            self.signalFrame.emit(frame, frame, 0)
            while True:
                if not self.work:
                    if self.capture.isOpened():
                        self.capture.release()
                        print("释放")
                    self.signalFrame.emit(None, 1, 0)
                    break
                time.sleep(0.01)
                success, frame = self.capture.read()
                if success:
                    orig, image, count = self.people_detect.detectVideo(frame)
                    if count > 0:
                        text = "行人碰撞预警({}人)".format(count)
                        orig = self.drawText(text, orig)
                    self.signalFrame.emit(frame, orig, count)
                else:
                    self.capture.release()
                    self.signalFrame.emit(None, 0, 0)
                    break
        except Exception as err:
            print(err)
            self.signalFailed.emit(str(err))
