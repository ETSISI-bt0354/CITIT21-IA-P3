import pandas as pd
from upmmlp.mlp import train_and_test

csv = pd.read_csv("src/diabetes_binary_health_indicators_BRFSS2015.csv", sep=",")
seed = 10

hidden_num_options = [1, 3, 10]
hidden_size_options = [5, 10, 20, 100]
learning_rate_options = [1, 0.1, 0.01, 0.001]

for hidden_num in hidden_num_options:
    for hidden_size in hidden_size_options:
        for learning_rate in learning_rate_options:
            print(f"hidden_num: {hidden_num}, hidden_size: {hidden_size}, learning_rate: {learning_rate}, test error: {train_and_test(csv, seed, learning_rate, hidden_size, hidden_num)}")