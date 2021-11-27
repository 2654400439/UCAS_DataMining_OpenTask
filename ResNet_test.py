import numpy as np
import os
import tensorflow as tf
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


model = tf.keras.models.load_model('ResNet_best_val_acc.h5')
print(model.summary())

type = input('whole_test or single_test -> 1 or 2:')
if type == '1':
    test_datagen = ImageDataGenerator(rescale=1 / 255)
    test_generator = test_datagen.flow_from_directory(
        './images_background_bak2/data_unbalanced/val',
        target_size=(100, 26),
        color_mode='grayscale',
        batch_size=200
    )
    pre = model.predict_generator(
        test_generator,
        verbose=2
    )
    score = model.evaluate(
        test_generator,
        verbose=2
    )
elif type == '2':
    while True:
        label = input('star or galaxy or qso -> 1 or 2 or 3:')
        if label == '1':
            path = './images_background_bak2/data_unbalanced/val/star/'
        elif label == '2':
            path = './images_background_bak2/data_unbalanced/val/galaxy/'
        else:
            path = './images_background_bak2/data_unbalanced/val/qso/'
        num = input('which one?')
        dir = os.listdir(path)
        path += dir[int(num)]
        img = Image.open(path)
        img = np.array(img).astype(np.float64)
        img *= 1/255
        img = img.reshape(1,-1)
        img = np.delete(img,np.s_[-1],axis=1)
        print(model.predict(
            np.array([np.array([img.reshape(100,26)]).reshape(100,26,1)])
        ))
else:
    print('error')


