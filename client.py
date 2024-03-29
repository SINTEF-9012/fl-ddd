import argparse
import os

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Do not consume all GPU at once
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "True"

import flwr as fl
import tensorflow as tf

from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator as data_augment

# Parse arguments
parser = argparse.ArgumentParser(description="DDD client")
parser.add_argument(
    "--client",
    required=True,
    help="Partition of the dataset (from A to ZC). "
    "The dataset is divided into 28 partitions.",
)
args = parser.parse_args()

#data augmetation
data_generate_training = data_augment(rescale=1./255,
                              shear_range = 0.2,
                              zoom_range = 0.2,
                              fill_mode = "nearest",
                              horizontal_flip = True,
                              width_shift_range = 0.2,
                              height_shift_range = 0.2,
                              validation_split = 0.15)

# the path for client A should be something like "/home/user/ddd/A"
datadir= "REPLACE_WITH_THE_PATH_WHERE_YOU_EXTRACTED_THE_DDD_DATASET" + "/" + args.client

#data split and loading
traind = data_generate_training.flow_from_directory(datadir,
                                          target_size = (227, 227),
                                          #seed = 123,
                                          shuffle=True,
                                          batch_size = 32,
                                          subset = "training")

testd = data_generate_training.flow_from_directory(datadir,
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

# compile model
CNNmodel.compile(optimizer='adam',
              loss="binary_crossentropy",
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

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

        loss, accuracy, precision, recall = self.model.evaluate(self.testd)
        return loss, len(self.testd), {"accuracy": accuracy, "precision": precision, "recall": recall}

# Create Flower client
client = CifarClient(CNNmodel, traind, testd)

# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
