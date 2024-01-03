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
    
    return None
