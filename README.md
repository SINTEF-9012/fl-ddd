# An example to demonstrate the use of Federated Learning with non-IID data Flower Example using TensorFlow/Keras

This project contains two types of experiments:

Using Convolutional Neural Networks (CNNs) for driver drowsiness detection is a powerful application of deep learning. Among other tools, TensorFlow in combination with Keras provide a robust framework for implementing such systems. CNNs excel in image recognition tasks by automatically learning hierarchical features from visual data. In this context, facial images captured by in-vehicle cameras serve as input. TensorFlow, an open-source ML library, seamlessly integrates with Keras, a high-level neural networks API, simplifying the implementation of CNN architectures. By structuring layers with convolutional and pooling operations, a CNN can effectively extract intricate patterns and features from facial expressions, eye movements, and other indicators of drowsiness. Training the model on labelled datasets allows it to learn and generalise patterns associated with drowsiness. Once trained, the CNN can be deployed in real-time driver monitoring systems, providing accurate and rapid detection of drowsiness, thereby contributing to enhanced road safety.
## Data preparation

Drowsiness poses a significant challenge in various professions, where its impact can have severe consequences. Defined by feelings of sleepiness, fatigue, and reduced alertness, drowsiness compromises the ability to maintain focus, make quick decisions, and respond rapidly - all critical aspects of safe driving. In driving, drowsiness is a crucial concern due to its direct correlation with an increased risk of accidents. Insufficient or poor-quality sleep, long working hours, night shifts, and monotonous driving conditions contribute to drowsiness among drivers. Certain professions are at a higher risk of drowsiness-related incidents, particularly those involving extended hours on the road. Long-haul truck drivers, delivery professionals, and emergency service providers working irregular hours face heightened risks. Addressing drowsiness is vital for professions requiring driving to ensure safety on the roads. The implications of drowsy driving extend beyond individual performance, emphasising the need for effective strategies and technology-driven solutions to safeguard the well-being of drivers and the broader public on the road.

1. Download the Drowsiness Detection Dataset (DDD) from [Kaggle](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd/data)
2. Extract the DDD dataset into a folder (e.g. "/home/user/DDD/ALL"). This folder should contain two classes - "Drowsy" and "Not Drowsy".
3. Manually copy individual subjects to separate folders (e.g. "/home/user/ddd/A" for subject A, "/home/user/ddd/B" for subject B, and so on). All these folders should also contain two classes - "Drowsy" and "Not Drowsy".

## Centralised ML

In the centralised setup, the entire dataset of 41,790 images is accessible in one location, and the model is trained on this comprehensive dataset. The CNN iteratively learns to recognise patterns associated with drowsiness by adjusting its parameters based on the entire dataset. This centralised approach is straightforward, as it involves a single training process on the complete dataset.

Run the centralised ML training (100 epochs) using the following command:

```shell
python3 center.py
```

## Federated Learning

The centralised setup raises concerns related to data privacy, since the images contain identifiable personal information. Additionally, the centralised model may not generalise well to diverse driving conditions or individual differences in facial expressions, as it is trained on a uniform dataset. In the FL setup where the entire dataset is split into 28 separate parts, each representing a specific subject, the focus extends beyond individual model training to the aggregation of these decentralised model updates. After each local model is trained on its respective subset, the model updates are shared and aggregated to create a global model that captures the collective knowledge learned from all subjects. As the underlying FL framework, we used Flower, which we coupled with the centralised Tensorflow/Keras implementation. To make the comparison fair, we aimed for 10 training rounds 10 epochs each, thus resulting in 100 epochs in total.

You can run a bash script which will start the FL server and 28 separate clients:

```shell
bash ./run.sh
```

Alternatively, you can launch the FL server and 28 clients in separate terminals.

### Logs

You can inspect the logs in the "logs" folder to track the learning progress and the resulting model performance


