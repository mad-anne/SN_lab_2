import numpy as np


def _feed_forward_layer(inputs, weights, act_func):
    return act_func.get_output(np.dot(inputs, weights))


class MultiLayerPerceptron:
    def __init__(self, features, classes, hidden_neurons, learning_rate, act_func):
        self.features = features
        self.classes = classes
        self.hidden_neurons = hidden_neurons
        self.learning_rate = learning_rate
        self.act_func = act_func
        self.bias = np.ones((1, 1))
        self.weights_1 = 2 * np.random.rand(self.features + 1, self.hidden_neurons) - 1
        self.weights_2 = 2 * np.random.rand(self.hidden_neurons + 1, self.classes) - 1

    def predict(self, data):
        outputs = self._predict(data)
        return np.argmax(outputs[-1])

    def learn(self, train_set, epochs):
        data_set = train_set
        for _ in range(epochs):
            np.random.shuffle(data_set)
            self._learn_epoch(data_set)

    def init_weights(self, deviation):
        self.weights_1 = 2 * deviation * np.random.rand(self.features + 1, self.hidden_neurons) - deviation
        self.weights_2 = 2 * deviation * np.random.rand(self.hidden_neurons + 1, self.classes) - deviation

    def _learn_epoch(self, data_set):
        for data in data_set:
            self._backpropagate(self._predict(data.data), data.output, data.data)

    def _backpropagate(self, outputs, target, inputs):
        inputs = np.column_stack((inputs, self.bias))
        output = np.column_stack((outputs[-2], self.bias))

        l2_delta = (target - outputs[-1]) * (outputs[-1] * (1 - outputs[-1]))
        l1_delta = np.dot(l2_delta, self.weights_2.transpose()) * (output * (1 - output))
        l1_delta = np.reshape(np.delete(l1_delta, l1_delta.shape[1] - 1), (1, l1_delta.shape[1] - 1))

        self.weights_2 += np.dot(output.transpose(), l2_delta)
        self.weights_1 += np.dot(inputs.transpose(), l1_delta)

    def _predict(self, data):
        output_hidden = _feed_forward_layer(np.column_stack((data, self.bias)), self.weights_1, self.act_func)
        output = _feed_forward_layer(np.column_stack((output_hidden, self.bias)), self.weights_2, self.act_func)
        return output_hidden, output

    def validate(self, data_set):
        return sum([data.label == self.predict(data.data) for data in data_set]) / len(data_set)
