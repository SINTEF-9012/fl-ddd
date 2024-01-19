import os

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Do not consume all GPU at once
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "True"

from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator as data_augment
from keras.models import load_model
from keras.layers import Input

#data augmetation
data_generate_training = data_augment (rescale=1./255,
                              shear_range = 0.2,
                              zoom_range = 0.2,
                              fill_mode = "nearest",
                              horizontal_flip = True,
                              width_shift_range = 0.2,
                              height_shift_range = 0.2,
                              validation_split = 0.15)

# the path for the aggregator should be something like "/home/user/ddd/"
datadir= "REPLACE_WITH_THE_PATH_WHERE_YOU_EXTRACTED_THE_DDD_DATASET"

#data split and loading
traind = data_generate_training.flow_from_directory(datadir,
                                          target_size = (227, 227),
                                          seed = 123,
                                          batch_size = 32,
                                          subset = "training")

testd = data_generate_training.flow_from_directory(datadir,
                                          target_size = (227, 227),
                                          seed = 123,
                                          batch_size = 32,
                                          subset = "validation")

#Building Model
CNNmodel = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), input_shape=(227, 227, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.5),

    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.5),

    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.5),

    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128, activation = 'relu', kernel_regularizer='l1'),
    keras.layers.Dense(2, activation = 'sigmoid')
])

#compile model
CNNmodel.compile(optimizer='adam',
              loss="binary_crossentropy",
              metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])

#train
history = CNNmodel.fit(traind, epochs = 100, validation_data = testd)