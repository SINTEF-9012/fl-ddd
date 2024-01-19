# An exmaple to demonstrate the use of Federated Learning with non-IID data Flower Example using TensorFlow/Keras

This introductory example to Flower uses Keras but deep knowledge of Keras is not necessarily required to run the example. However, it will help you understanding how to adapt Flower to your use-cases.
Running this example in itself is quite easy.

## Data preparation

Drowsiness poses a significant challenge in various professions, where its impact can have severe consequences. Defined by feelings of sleepiness, fatigue, and reduced alertness, drowsiness compromises the ability to maintain focus, make quick decisions, and respond rapidly - all critical aspects of safe driving. In driving, drowsiness is a crucial concern due to its direct correlation with an increased risk of accidents. Insufficient or poor-quality sleep, long working hours, night shifts, and monotonous driving conditions contribute to drowsiness among drivers. Certain professions are at a higher risk of drowsiness-related incidents, particularly those involving extended hours on the road. Long-haul truck drivers, delivery professionals, and emergency service providers working irregular hours face heightened risks. Addressing drowsiness is vital for professions requiring driving to ensure safety on the roads. The implications of drowsy driving extend beyond individual performance, emphasising the need for effective strategies and technology-driven solutions to safeguard the well-being of drivers and the broader public on the road.

1. Download the Drowsiness Detection Dataset (DDD) from [Kaggle](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd/data)
2. Extract the DDD dataset into a folder (e.g. "/home/user/DDD/ALL"). This folder should contain two classes - "Drowsy" and "Not Drowsy".
3. Manually copy individual subjects to separate folders (e.g. "/home/user/ddd/A" for subject A, "/home/user/ddd/B" for subject B, and so on). All these folders should also contain two classes - "Drowsy" and "Not Drowsy".

## Centralised ML

Run the centralised ML training (100 epochs) using the following command:

```shell
python3 center.py
```

## Federated Learning

You can run a bash script which will start the FL server and 28 separate clients:

```shell
bash ./run.sh
```

Alternatively, you can launch the FL server and 28 clients in separate terminals.

### Logs

You can inspect the logs in the "logs" folder to track the learning progress and the resulting model performance


