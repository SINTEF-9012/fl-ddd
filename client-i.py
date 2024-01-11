import flwr as fl
#import tensorflow as tf

#from pathlib import Path
import os
#import numpy as np
#import pandas as pd

from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator as data_augment
from keras.optimizers import SGD
#from keras.models import load_model
#from keras.layers import Input,Conv2D,MaxPooling2D,Dropout,Flatten,Dense,GlobalAveragePooling2D,BatchNormalization
#from keras.callbacks import EarlyStopping,ModelCheckpoint

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

#data augmetation
data_generate_training = data_augment (rescale=1./255,
                              shear_range = 0.2,
                              zoom_range = 0.2,
                              fill_mode = "nearest",
                              horizontal_flip = True,
                              width_shift_range = 0.2,
                              height_shift_range = 0.2,
                              validation_split = 0.15)

data_generate_test = data_augment(rescale = 1./255)

#data preprocessing and augmentation
traind = data_generate_training.flow_from_directory("I",
                                          target_size = (227, 227),
                                          #seed = 123,
                                          shuffle=True,
                                          batch_size = 32,
                                          subset = "training")

testd = data_generate_training.flow_from_directory("I",
                                          target_size = (227, 227),
                                          #seed = 123,
                                          shuffle=True,
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

opt = SGD(lr=0.001)
CNNmodel.compile(optimizer=opt,
              loss="binary_crossentropy",
              metrics=['accuracy'])

#history = CNNmodel.fit(traind, epochs = 1, validation_data = testd)

# print(CNNmodel.get_weights())

# Load model and data (MobileNetV2, CIFAR-10)
# model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
# model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


# Define Flower client
class CifarClient(fl.client.NumPyClient):
    
    def __init__(self, model, traind, testd):
        self.model = model
        self.traind = traind
        self.testd = testd
    
    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(x=self.traind, epochs=1)
        return self.model.get_weights(), len(self.traind), {}

    def evaluate(self, parameters, config):
        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        # steps: int = config["val_steps"]

        loss, accuracy = self.model.evaluate(self.testd) #testd
        return loss, len(self.testd), {"accuracy": accuracy}

# Create Flower client
client = CifarClient(CNNmodel, traind, testd)

# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
