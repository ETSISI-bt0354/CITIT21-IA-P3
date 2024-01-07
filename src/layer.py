import numpy as np
import numpy.typing as npt

class layer():
    def __init__(self, input: int, size: int, rng):
        self.weight = rng.random((input, size)) * 2
        self.bias = rng.random((1, size)) * 2
