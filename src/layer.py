import numpy as np
import numpy.typing as npt

class layer:
    def __init__(self, input: int, size: int):
        self.weight = np.random.rand(input, size)
        self.bias = np.random.rand(1, size)

    def foward_propagation(self, input: npt.NDArray[float]) -> npt.NDArray[float]:
        output = np.dot(input, self.weight) + self.bias
        return activation(output)

    def backward_propagation(self, gradient: npt.NDArray[float], input: npt.NDArray, learning_rate: float):
        delta_weight = learning_rate * np.dot(np.atleast_2d(input).T, gradient)
        delta_bias = learning_rate * gradient

        return delta_weight, delta_bias

def activation(output: npt.NDArray) -> npt.NDArray:
    return 1 / (1 + np.exp(-output))

def gradient(loss: npt.NDArray, output: npt.NDArray) -> npt.NDArray:
    return loss * output * (1 - output)