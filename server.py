import os

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Do not consume all GPU at once
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "True"

from typing import List, Tuple, Dict, Optional

import flwr as fl
from flwr.common import Metrics

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

# the path should be something like "/home/user/ddd/"
datadir= "REPLACE_WITH_THE_PATH_WHERE_YOU_EXTRACTED_THE_DDD_DATASET"

#data preprocessing and augmentation
traind = data_generate_training.flow_from_directory(datadir,
                                          target_size = (227, 227),
                                          seed = 123,
                                          batch_size = 32,
                                          subset = "training")

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

#Compile model
CNNmodel.compile(optimizer='adam',
              loss="binary_crossentropy",
              metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])

#Define aggregated evaluation fucntion
def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""
    testd = data_generate_training.flow_from_directory("data/ALL",
                                          target_size = (227, 227),
                                          #seed = 123,
                                          batch_size = 32,
                                          subset = "validation")

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        
        model.set_weights(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(testd, batch_size=32)
        return loss, {"accuracy": accuracy}

    return evaluate

# Define metric aggregation function - Weighted Average
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy, precision and recall of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    precisions = [num_examples * m["precision"] for num_examples, m in metrics]
    recalls = [num_examples * m["recall"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples), "precision": sum(precisions) / sum(examples), "recall": sum(recalls) / sum(examples)}

# Create strategy
strategy = fl.server.strategy.FedAvg(
    min_fit_clients=28,
    min_evaluate_clients=28,
    min_available_clients=28,
    evaluate_fn=get_evaluate_fn(CNNmodel),
    initial_parameters=fl.common.ndarrays_to_parameters(CNNmodel.get_weights()),
    evaluate_metrics_aggregation_fn=weighted_average
)

# Start Flower server 
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=1),
    strategy=strategy
)