import cv2 as cv
import os
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

X = 1
IMG_WIDTH = 180
IMG_HEIGHT = 180
Data_path = os.path.join('..', 'Data', 'data')
trained_mode_path = os.path.join('..', 'Pre_Trained', 'pre-trained.h5')
pet_net_path = os.path.join('..', 'Normal', 'PetNet.h5')

# read the image
images = []
for img_path in os.listdir(Data_path):
    print(os.path.join(Data_path, img_path))
    image = cv.imread(os.path.join(Data_path, img_path))

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    image = cv.resize(image, (IMG_HEIGHT, IMG_WIDTH))

    images.append(image)

images = np.array(images, dtype = 'float32')/255.0
print(images.shape)

# load the models
PreTrained = tf.keras.models.load_model(trained_mode_path)
PetNet = tf.keras.models.load_model(pet_net_path)

# predict the image
p1 = PreTrained.predict(images)
p2 = PetNet.predict(images)
result1 = np.squeeze((np.where(p1 >= 0.5, 'cat', 'dog')))
result2 = np.squeeze((np.where(p2 >= 0.5, 'cat', 'dog')))

print(result1)
print("-----------")
print(result2)
