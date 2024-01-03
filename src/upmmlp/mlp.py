import pandas as pd
import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import MinMaxScaler

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
    return len(train) + len(test) + len(validation)


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