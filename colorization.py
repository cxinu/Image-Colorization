import numpy as np
import cv2


class Colorize:
    def __init__(self) -> None:
        self.net = cv2.dnn.readNetFromCaffe(
            './model/colorization_deploy_v2.prototxt', './model/colorization_release_v2.caffemodel')
        self.pts = np.load('./model/pts_in_hull.npy')
        self.pts = self.pts.transpose().reshape(2, 313, 1, 1)

        class8 = self.net.getLayerId("class8_ab")
        conv8 = self.net.getLayerId("conv8_313_rh")
        self.net.getLayer(class8).blobs = [self.pts.astype("float32")]
        self.net.getLayer(conv8).blobs = [np.full(
            [1, 313], 2.606, dtype='float32')]

    def colorize(self, img_path):
        self.image = cv2.imread(img_path)
        scaled = self.image.astype("float32")/255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50

        self.net.setInput(cv2.dnn.blobFromImage(L))
        ab = self.net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab = cv2.resize(ab, (self.image.shape[1], self.image.shape[0]))
        L = cv2.split(lab)[0]

        self.colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
        self.colorized = cv2.cvtColor(self.colorized, cv2.COLOR_LAB2BGR)
        self.colorized = np.clip(self.colorized, 0, 1)
        self.colorized = (255 * self.colorized).astype("uint8")

    def show(self):
        cv2.imshow("Original", self.image)
        cv2.imshow("Colorized", self.colorized)
        cv2.waitKey(0)


if __name__ == "__main__":
    image = "mastani"
    AI = Colorize()
    AI.colorize(img_path=f"images/{image}.jpg")
    AI.show()
