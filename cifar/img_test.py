import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train.astype(np.float32)
x_train = x_train / 255.0

plt.figure("ok")
img1 = x_train[0,:,:,:]
print(type(img1))
print(img1)
plt.imshow(img1)
# cv2.imshow('ss',img1)
# input("Ok")