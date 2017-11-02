import numpy as np


def _feed_forward_layer(inputs, weights, act_func):
    return act_func.get_output(np.dot(inputs, weights))


class MultiLayerPerceptron:
    def __init__(self, features, classes, hidden_neurons, learning_rate, act_func):
        self.features = features
        self.classes = classes
        self.hidden_neurons = hidden_neurons
        self.learning_rate = learning_rate
        self.weights_1 = 2 * np.random.rand(features + 1, hidden_neurons) - 1
        self.weights_2 = 2 * np.random.rand(hidden_neurons + 1, classes) - 1
        self.act_func = act_func
        self.bias = np.ones((1, 1))

    def learn(self, data_set, epochs):
        for _ in range(epochs):
            self._learn_epoch(data_set)

    def _learn_epoch(self, data_set):
        for data in data_set:
            pass

    def _get_error(self, result, target):
        return np.multiply(np.power(np.subtract(target, result), 2), 0.5)

    def predict(self, data):
        outputs = self._predict(data)
        return np.argmax(outputs[-1])

    def _predict(self, data):
        output_hidden = _feed_forward_layer(np.column_stack((data, self.bias)), self.weights_1, self.act_func)
        output = _feed_forward_layer(np.column_stack((output_hidden, self.bias)), self.weights_2, self.act_func)
        return output_hidden, output
