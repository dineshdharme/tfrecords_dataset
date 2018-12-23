import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras import optimizers
from tensorflow.keras import applications
from tensorflow.keras.models import Model
import  glob
from tfrecords_utils import create_dataset


# dimensions of our images.
img_width, img_height = 150, 150





# Small Conv Net
# a simple stack of 3 convolution layers with a ReLU activation and followed by max-pooling layers.
model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=(img_width, img_height,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

#Training

train_tfrecord = 'data/processed/tfrecords/train/'
test_tfrecord = 'data/processed/tfrecords/validation/'

train_paths = glob.glob(os.path.join(train_tfrecord, '*'))
test_paths = glob.glob(os.path.join(test_tfrecord, '*'))

train_dataset  = create_dataset(train_paths)
test_dataset  = create_dataset(test_paths)



model.fit(
        train_dataset.make_one_shot_iterator(),
        steps_per_epoch=5,
        epochs=10,
        shuffle=True,
        validation_data=test_dataset.make_one_shot_iterator(),
        validation_steps=2,
        verbose=1)


#About 60 seconds an epoch when using CPU

model.save_weights('models/basic_cnn_30_epochs.h5')
