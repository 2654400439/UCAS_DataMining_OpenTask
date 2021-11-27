import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

model = tf.keras.models.Sequential()
model.add(keras.layers.Conv2D(32,(3,3),padding='SAME',activation='relu',input_shape=(51,51,1)))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(64,(3,3),padding='SAME',activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(128,(3,3),padding='SAME',activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(128,activation='relu'))
model.add(keras.layers.Dense(1024,activation='relu'))
model.add(keras.layers.Dense(3,activation='softmax'))
model.compile(loss=keras.losses.sparse_categorical_crossentropy,optimizer=keras.optimizers.RMSprop(lr=0.001),metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1/255,validation_split=0.5)
train_generator = train_datagen.flow_from_directory(
    './data_unbalanced/train',
    target_size=(51,51),
    color_mode='grayscale',
    batch_size=50,
    class_mode='sparse'
)
validation_datagen = ImageDataGenerator(rescale=1/255,validation_split=0.5)
validation_generator = validation_datagen.flow_from_directory(
    './data_unbalanced/val',
    target_size=(51,51),
    color_mode='grayscale',
    batch_size=50,
    class_mode='sparse'
)
model_name = 'model_ex-{epoch:03d}_acc-{val_accuracy:03f}.h5'
trained_model_dir='./model/'
model_path = os.path.join(trained_model_dir, model_name)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
             filepath=model_path,
             monitor='val_accuracy',
            verbose=1,
            save_weights_only=True,
            save_best_only=True,
            mode='max',
            period=1)

history = model.fit_generator(
    train_generator,
    epochs=30,
    verbose=1,
    validation_data=validation_generator,
    callbacks=[checkpoint]
)
model.save('CNN_paras_30_epochs.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(len(acc))
plt.plot(epochs,acc,'r',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and Validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()
