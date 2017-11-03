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
        self.errors_sum = 0
        self.last_weights_1_update = np.zeros((self.features + 1, self.hidden_neurons))
        self.last_weights_2_update = np.zeros((self.hidden_neurons + 1, self.classes))

    def predict(self, data):
        outputs = self._predict(data)
        return np.argmax(outputs[-1])

    def learn(self, train_set, epochs, min_mse, momentum):
        data_set = train_set
        for epoch in range(epochs):
            self.errors_sum = 0
            np.random.shuffle(data_set)
            self._learn_epoch(data_set, momentum)
            mse = self.errors_sum / len(data_set)
            if self._has_stop_criterion_met(mse, min_mse):
                return epoch + 1
        return epochs

    def validate(self, data_set):
        return sum([data.label == self.predict(data.data) for data in data_set]) / len(data_set)

    def init_weights(self, deviation):
        self.weights_1 = 2 * deviation * np.random.rand(self.features + 1, self.hidden_neurons) - deviation
        self.weights_2 = 2 * deviation * np.random.rand(self.hidden_neurons + 1, self.classes) - deviation

    def _learn_epoch(self, data_set, momentum):
        self.last_weights_1_update = np.zeros((self.features + 1, self.hidden_neurons))
        self.last_weights_2_update = np.zeros((self.hidden_neurons + 1, self.classes))

        for data in data_set:
            self._backpropagate(self._predict(data.data), data.output, data.data, momentum)

    def _backpropagate(self, outputs, target, inputs, momentum):
        self.errors_sum += sum(np.power(outputs[-1] - target, 2)[0])

        inputs = np.column_stack((inputs, self.bias))
        output = np.column_stack((outputs[-2], self.bias))

        l2_delta = (target - outputs[-1]) * (outputs[-1] * (1 - outputs[-1]))
        l1_delta = np.dot(l2_delta, self.weights_2.transpose()) * (output * (1 - output))
        l1_delta = np.reshape(np.delete(l1_delta, l1_delta.shape[1] - 1), (1, l1_delta.shape[1] - 1))

        self.last_weights_2_update = np.add(
            self.learning_rate * np.dot(output.transpose(), l2_delta),
            momentum * self.last_weights_2_update
        )
        self.last_weights_1_update = np.add(
            self.learning_rate * np.dot(inputs.transpose(), l1_delta),
            momentum * self.last_weights_1_update
        )

        self.weights_2 += self.last_weights_2_update
        self.weights_1 += self.last_weights_1_update

    def _predict(self, data):
        output_hidden = _feed_forward_layer(np.column_stack((data, self.bias)), self.weights_1, self.act_func)
        output = _feed_forward_layer(np.column_stack((output_hidden, self.bias)), self.weights_2, self.act_func)
        return output_hidden, output

    def _has_stop_criterion_met(self, mse, min_mse):
        return mse <= min_mse
