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

    train, test, validation = partition(dataset, 0.9, 0.05, 0.05)
    aux = []

    network = []
    if (hidden_num >= 1):
        network.append(ly.layer(train.shape[1] - 1, hidden_size))

    for _ in range(hidden_num - 1):
        network.append(ly.layer(hidden_size, hidden_size))

    network.append(ly.layer(hidden_size, 1))

    best_f1_score = 0
    cyces_without_improvement = 0
    while cyces_without_improvement < 5:
        train_epoch(train, network, learning_rate)
        current_f1_score = validation_epoch(validation, network)
        aux.append(current_f1_score)

        if current_f1_score > best_f1_score:
            best_f1_score = current_f1_score
            cyces_without_improvement = 0
        else:
            cyces_without_improvement += 1

    return aux



def partition(data: pd.DataFrame, train_percentage: float, test_percentage: float, validation_percentage: float) -> tuple[npt.NDArray[float], npt.NDArray[float], npt.NDArray[float]]:
    rng = np.random.default_rng()

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
    diabetes_validation = diabetes_data_norm[int(n_diabetes * (train_percentage + test_percentage)):]

    non_diabetes_train = non_diabetes_data_norm[:int(n_non_diabetes * train_percentage)]
    non_diabetes_test = non_diabetes_data_norm[int(n_non_diabetes * train_percentage):int(n_non_diabetes * (train_percentage + test_percentage))]
    non_diabetes_validation = non_diabetes_data_norm[int(n_non_diabetes * (train_percentage + test_percentage)):]

    train = np.vstack((diabetes_train, non_diabetes_train))
    test = np.vstack((diabetes_test, non_diabetes_test))
    validation = np.vstack((diabetes_validation, non_diabetes_validation))

    rng.shuffle(train)
    rng.shuffle(test)
    rng.shuffle(validation)

    return train, test, validation

def loss_function(expected_value: npt.NDArray, network_output: npt.NDArray) -> npt.NDArray:
    return expected_value - network_output

def full_foward_propagation(input, network: list[ly.layer]):
    Y = [input]
    for layer in network:
        input = layer.foward_propagation(input)
        Y.append(input)

    return Y

def train_step(input, network, learning_rate):
    Y = full_foward_propagation(input[1:], network)

    loss = loss_function(input[0], Y[-1])

    deltas = []
    for index, layer in enumerate(reversed(network)):
        index = len(network) - index - 1
        gradient = ly.gradient(loss, Y[index + 1])

        deltas.append(layer.backward_propagation(gradient, Y[index], learning_rate))

        loss = np.dot(gradient, np.atleast_2d(layer.weight).T)

    for layer, (delta_weight, delta_bias) in zip(network, reversed(deltas)):
        layer.weight += delta_weight
        layer.bias += delta_bias
def train_epoch(train_data, network, learning_rate):
    for row in train_data:
        train_step(row, network, learning_rate)

def validation_epoch(validation_data, network) -> float:
    confusion_matrix = [[0, 0], [0, 0]]

    for row in validation_data:
        probability = full_foward_propagation(row[1:], network)[-1]
        prediction = round(probability)

        confusion_matrix[int(row[0])][prediction] += 1

    return F1_score(confusion_matrix[1][1], confusion_matrix[0][1], confusion_matrix[1][0])

def F1_score(true_positive: int, false_positive: int, false_negative: int) -> float:
    p = precision(true_positive, false_positive)
    r = recall(true_positive, false_negative)

    return 2 * p * r / (p + r)

def precision(true_positive: int, false_positive: int) -> float:
    return true_positive / (true_positive + false_positive)

def recall(true_positive: int, false_negative: int) -> float:
    return true_positive / (true_positive + false_negative)