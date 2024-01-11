import argparse
import os

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import flwr as fl
import tensorflow as tf

from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator as data_augment
from keras.optimizers import SGD

# import logging
# import sys

# Parse arguments
parser = argparse.ArgumentParser(description="DDD client")
parser.add_argument(
    "--client",
    required=True,
    help="Partition of the dataset (from A to ZC). "
    "The dataset is divided into 28 partitions.",
)
args = parser.parse_args()

# root = logging.getLogger()
# root.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# fileHandler = logging.FileHandler(args.client + ".log")
# fileHandler.setFormatter(formatter)
# root.addHandler(fileHandler)

# tf_log = tf.get_logger()
# tf_log.setLevel(logging.INFO)
# tf_fileHandler = logging.FileHandler(args.client + "_tf.log")
# tf_fileHandler.setFormatter(formatter)
# tf_log.addHandler(tf_fileHandler)

#consoleHandler = logging.StreamHandler()
#consoleHandler.setFormatter(logFormatter)
#rootLogger.addHandler(consoleHandler)

#data augmetation
data_generate_training = data_augment(rescale=1./255,
                              shear_range = 0.2,
                              zoom_range = 0.2,
                              fill_mode = "nearest",
                              horizontal_flip = True,
                              width_shift_range = 0.2,
                              height_shift_range = 0.2,
                              validation_split = 0.15)

#data_generate_test = data_augment(rescale = 1./255)

#data preprocessing and augmentation
traind = data_generate_training.flow_from_directory("data/" + args.client,
                                          target_size = (227, 227),
                                          #seed = 123,
                                          shuffle=True,
                                          batch_size = 32,
                                          subset = "training")

testd = data_generate_training.flow_from_directory("data/" + args.client,
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

# opt = SGD(lr=0.001)
CNNmodel.compile(optimizer='adam', # opt
              loss="binary_crossentropy",
              metrics=['accuracy'])

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
        self.model.fit(x=self.traind, epochs=10)
        return self.model.get_weights(), len(self.traind), {}

    def evaluate(self, parameters, config):
        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        # steps: int = config["val_steps"]

        loss, accuracy = self.model.evaluate(self.testd)
        return loss, len(self.testd), {"accuracy": accuracy}

# Create Flower client
client = CifarClient(CNNmodel, traind, testd)

# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
