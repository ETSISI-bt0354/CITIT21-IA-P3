import pandas as pd
import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import MinMaxScaler

from src import layer as ly


def train_and_test(dataset, seed=0, learning_rate=.0, hidden_size=0, hidden_num=0):
    """
    Returns the test metric after training with a partition of the dataset
    :param dataset: pandas dataset to train the network with
    :param seed: (int) used to initialize the random number generator
    :param learning_rate: (float) learning rate of the MLP
    :param hidden_size: (int) number of neurons in each hidden layer
    :param hidden_num: (int) number of hidden layers in the network
    :return: the testing error, a float number
    """

    # 1. Preprocess the dataset (normalization, randomization, etc.)
    # 2. Split the dataset into training, validation and testing
    # 3. Train the network
    # - 3.1. Initialize the network
    # - 3.2. Perform a training epoch
    # --- 3.2.1. Show the network a training sample
    # --- 3.2.2. Calculate y
    # --- 3.2.3. Calculate the loss (d-y)
    # --- 3.2.4. Calculate the weight updates for each layer using backpropagation
    # --- 3.2.5. Update the weights of each layer
    # - 3.3. Perform a validation epoch
    # - 3.4. Calculate train and validation metrics
    # - 3.5. Repeat 3.2-3.4 until the stopping criterion is met
    # 4. Test the network

    rng = np.random.default_rng(seed)

    train, test, validation = partition(dataset, 0.9, 0.05, 0.05, rng)

    network = create_network(train.shape[1] - 1, hidden_num, hidden_size, rng)

    batch_size = 1

    best_f1_score = 0
    cycles_without_improvement = 0

    while cycles_without_improvement < 5:
        train_epoch(train, network, learning_rate, batch_size)
        try:
            current_f1_score = validation_epoch(validation, network)
            if current_f1_score > best_f1_score:
                best_f1_score = current_f1_score
                cycles_without_improvement = 0
            else:
                cycles_without_improvement += 1
        except:
            pass

    return validation_epoch(test, network)

def partition(data: pd.DataFrame, train_percentage: float, test_percentage: float, validation_percentage: float, rng) -> tuple[npt.NDArray[float], npt.NDArray[float], npt.NDArray[float]]:
    mms = MinMaxScaler()
    data_norm = pd.DataFrame(mms.fit_transform(data), columns=data.columns)

    data_norm_grouped = data_norm.groupby(data_norm["Diabetes_binary"])

    diabetes_data_norm = data_norm_grouped.get_group(1.0).to_numpy()
    non_diabetes_data_norm = data_norm_grouped.get_group(0.0).to_numpy()

    rng.shuffle(diabetes_data_norm)
    rng.shuffle(non_diabetes_data_norm)

    n_diabetes = len(diabetes_data_norm)
    n_non_diabetes = len(non_diabetes_data_norm)

    diabetes_train = diabetes_data_norm[:int(n_diabetes * train_percentage)]
    diabetes_test = diabetes_data_norm[int(n_diabetes * train_percentage):int(n_diabetes * (train_percentage + test_percentage))]
    diabetes_validation = diabetes_data_norm[int(n_diabetes * (train_percentage + test_percentage)):int(n_diabetes * (train_percentage + test_percentage + validation_percentage))]

    non_diabetes_train = non_diabetes_data_norm[:int(n_non_diabetes * train_percentage)]
    non_diabetes_test = non_diabetes_data_norm[int(n_non_diabetes * train_percentage):int(n_non_diabetes * (train_percentage + test_percentage))]
    non_diabetes_validation = non_diabetes_data_norm[int(n_non_diabetes * (train_percentage + test_percentage)):int(n_non_diabetes * (train_percentage + test_percentage + validation_percentage))]

    train = np.vstack((diabetes_train, non_diabetes_train))
    test = np.vstack((diabetes_test, non_diabetes_test))
    validation = np.vstack((diabetes_validation, non_diabetes_validation))

    rng.shuffle(train)
    rng.shuffle(test)
    rng.shuffle(validation)

    return train, test, validation


def create_network(input_size, hidden_num, hidden_size, rng):
    network = []
    if hidden_num >= 1:
        network.append(ly.layer(input_size, hidden_size, rng))

    for _ in range(hidden_num - 1):
        network.append(ly.layer(hidden_size, hidden_size, rng))

    if hidden_num >= 1:
        network.append(ly.layer(hidden_size, 1, rng))
    else:
        network.append(ly.layer(input_size, 1, rng))

    return network


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(sigmoid_output):
    return sigmoid_output * (1 - sigmoid_output)


def gradient(loss, activation_derivative):
    return loss * activation_derivative

def loss_function(expected_value: npt.NDArray, network_output: npt.NDArray) -> npt.NDArray:
    return expected_value - network_output


def train_step(data_input, network, learning_rate):
    Y = [data_input[:, 1:]]
    for layer in network:
        Y.append(sigmoid(np.dot(Y[-1], layer.weight) + layer.bias))

    loss = loss_function(np.atleast_2d(data_input[:, 0]).T, Y[-1])

    deltas = []
    for index, layer in enumerate(reversed(network)):
        index = len(network) - index - 1
        gradient_value = gradient(loss, sigmoid_derivative(Y[index + 1]))

        delta_weight = learning_rate * np.dot(np.atleast_2d(Y[index]).T, gradient_value)
        delta_bias = learning_rate * np.sum(gradient_value, axis=0, keepdims=True)

        deltas.append((delta_weight, delta_bias))

        loss = np.dot(gradient_value, np.atleast_2d(layer.weight).T)

    for layer, (delta_weight, delta_bias) in zip(network, reversed(deltas)):
        layer.weight += delta_weight
        layer.bias += delta_bias


def train_epoch(train_data, network, learning_rate, batch_size):
    for i in range(0, len(train_data), batch_size):
        batch = train_data[i:i + batch_size]
        train_step(batch, network, learning_rate)


def validation_epoch(validation_data, network) -> float:
    confusion_matrix = [[0, 0], [0, 0]]
    network_input = validation_data[:, 1:]

    for layer in network:
        network_input = sigmoid(np.dot(network_input, layer.weight) + layer.bia)

    for result, prediction in zip(validation_data[:, 0], network_input):
        confusion_matrix[int(result)][round(prediction.item())] += 1

    return F1_score(confusion_matrix[1][1], confusion_matrix[0][1], confusion_matrix[1][0])


def F1_score(true_positive: int, false_positive: int, false_negative: int) -> float:
    p = precision(true_positive, false_positive)
    r = recall(true_positive, false_negative)

    return 2 * p * r / (p + r)


def precision(true_positive: int, false_positive: int) -> float:
    return true_positive / (true_positive + false_positive)


def recall(true_positive: int, false_negative: int) -> float:
    return true_positive / (true_positive + false_negative)
