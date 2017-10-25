import random

from matrix import Matrix


class Layer:
    def __init__(self, inputs, size, act_func):
        self.act_func = act_func
        self.inputs = inputs
        self.size = size
        self.weights = Matrix(inputs, size)

    def init_random_weights(self, deviation):
        for i in range(self.inputs):
            for j in range(self.size):
                self.weights.set_value(i, j, random.uniform(-deviation, deviation))
