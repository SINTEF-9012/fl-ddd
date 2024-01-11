from typing import List, Tuple, Dict, Optional
import os

import flwr as fl
from flwr.common import Metrics

from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator as data_augment
from keras.models import load_model
from keras.layers import Input

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

#data preprocessing and augmentation
traind = data_generate_training.flow_from_directory("data/ALL",
                                          target_size = (227, 227),
                                          seed = 123,
                                          batch_size = 32,
                                          subset = "training")

#data_generate_test = data_augment(rescale = 1./255)

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

CNNmodel.compile(optimizer='adam',
              loss="binary_crossentropy",
              metrics=['accuracy'])

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

def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one local epoch,
    increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 1 if server_round < 4 else 2,
    }
    return config

def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five batches) during
    rounds one to three, then increase to ten local evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

# Create strategy
strategy = fl.server.strategy.FedAvg(
    #fraction_fit=0.3,
    #fraction_evaluate=0.2,
    min_fit_clients=28,
    min_evaluate_clients=28,
    min_available_clients=28,
    evaluate_fn=get_evaluate_fn(CNNmodel),
    #on_fit_config_fn=fit_config,
    #on_evaluate_config_fn=evaluate_config,
    initial_parameters=fl.common.ndarrays_to_parameters(CNNmodel.get_weights()),
    evaluate_metrics_aggregation_fn=weighted_average
)

# Start Flower server 
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=1),
    strategy=strategy
)