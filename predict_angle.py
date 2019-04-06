import numpy as np
from keras.models import load_model
import cv2


class AnglePredicter:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        
    def predict(self, image):
        image = image[180:380, :, :]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        image = cv2.GaussianBlur(image, (3, 3), 0)
        image = cv2.resize(image, (200, 66))
        image = image / 255
        image = np.array([image])
        steering_angle = float(self.model.predict(image))
        return steering_angle

angle_predicter = AnglePredicter("./MODELS/model-0.0417-0.0270-300steps-20epochs-100batch-cloudData.h5")

files = ["./images/image045.jpeg", "./images/image046.jpeg", "./images/image047.jpeg", "./images/image048.jpeg", "./images/image049.jpeg", "./images/image050.jpeg", "./images/image051.jpeg", "./images/image052.jpeg", "./images/image053.jpeg"]

for img_path in files:
    print(angle_predicter.predict(cv2.imread(img_path)))
