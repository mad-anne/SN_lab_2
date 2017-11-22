import numpy as np


class AutoEncoderMLP:
    """ Basic class for MLP network learned by back propagation """
    def __init__(self, features, hidden_neurons, act_func, epochs, learning_rate, deviation):
        self.features = features
        self.hidden_neurons = hidden_neurons
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.act_func = act_func
        self.bias = np.ones((1, 1))

        # init weights
        self.weights_1 = 2 * deviation * np.random.rand(self.features + 1, self.hidden_neurons) - deviation
        self.weights_2 = 2 * deviation * np.random.rand(self.hidden_neurons + 1, self.features) - deviation

    def predict(self, data):
        """ Returns calculated output pattern """
        return self._predict(data)[1]

    def learn(self, train_set, test_set):
        data_set = train_set
        train_errors = []
        test_errors = []
        for epoch in range(self.epochs):
            print(f'Learning epoch {epoch}')

            # learning phase
            np.random.shuffle(data_set)
            self._learn_epoch(data_set)

            # calculate current error on train and test sets
            train_errors.append(self.validate(train_set))
            test_errors.append(self.validate(test_set))
            print(f'Train error: {train_errors[-1]}')
            print(f'Test error: {test_errors[-1]}')
        return train_errors, test_errors

    def validate(self, data_set):
        """ Returns mean error in % from mean squared errors for every test input """
        return np.array([self._get_squared_error(data.data, self.predict(data.data)) * 100 for data in data_set]).mean()

    @staticmethod
    def _get_squared_error(expected_output, output):
        """ Returns mean squared error comparing  """
        return np.power(expected_output - output, 2).mean()

    def set_weights(self, weights_1, weights_2):
        self.weights_1 = weights_1
        self.weights_2 = weights_2

    def get_weights(self):
        return self.weights_1, self.weights_2

    def _learn_epoch(self, data_set):
        for data in data_set:
            self._backpropagate(data.data, *self._predict(data.data))

    def _backpropagate(self, inputs, output_hidden, output):
        # append bias to input and hidden layer
        layer_1 = np.column_stack((inputs, self.bias))
        layer_2 = np.column_stack((output_hidden, self.bias))

        # calculate weights delta for output and hidden layers
        l2_delta = (inputs - output) * (output * (1 - output))
        l1_delta = np.dot(l2_delta, self.weights_2.transpose()) * (layer_2 * (1 - layer_2))
        l1_delta = np.reshape(np.delete(l1_delta, l1_delta.shape[1] - 1), (1, l1_delta.shape[1] - 1))

        # update weights
        self.weights_2 += self.learning_rate * np.dot(layer_2.transpose(), l2_delta)
        self.weights_1 += self.learning_rate * np.dot(layer_1.transpose(), l1_delta)

    def _predict(self, data):
        """ Calculates output in hidden and output layer with forward propagation """
        output_hidden = self._feed_forward_layer(np.column_stack((data, self.bias)), self.weights_1)
        output = self._feed_forward_layer(np.column_stack((output_hidden, self.bias)), self.weights_2)
        return output_hidden, output

    def _feed_forward_layer(self, inputs, weights):
        return self.act_func.get_output(np.dot(inputs, weights))


class DropoutMLP(AutoEncoderMLP):
    def __init__(self, *args, **kwargs):
        self.probability = kwargs.pop('keep_prob')
        super().__init__(*args, **kwargs)

    def _learn_epoch(self, data_set):
        for data in data_set:
            dropout = np.random.rand(self.hidden_neurons, 1).transpose() < self.probability
            self._backpropagate(data.data, *self._predict(data.data, dropout=dropout))

    def _predict(self, data, dropout=None):
        output_hidden = self._feed_forward_layer(np.column_stack((data, self.bias)), self.weights_1)
        if dropout is not None:
            output_hidden = np.multiply(output_hidden, dropout) / self.probability
        output = self._feed_forward_layer(np.column_stack((output_hidden, self.bias)), self.weights_2)
        return output_hidden, output


class L2RegularizationMLP(AutoEncoderMLP):
    def __init__(self, *args, **kwargs):
        self.regularization_term = kwargs.pop('regularization_term')
        super().__init__(*args, **kwargs)

    def _backpropagate(self, inputs, output_hidden, output):
        # append bias to input and hidden layer
        layer_1 = np.column_stack((inputs, self.bias))
        layer_2 = np.column_stack((output_hidden, self.bias))

        # calculate weights delta for output and hidden layers
        l2_delta = (inputs - output) * (output * (1 - output))
        l1_delta = np.dot(l2_delta, self.weights_2.transpose()) * (layer_2 * (1 - layer_2))
        l1_delta = np.reshape(np.delete(l1_delta, l1_delta.shape[1] - 1), (1, l1_delta.shape[1] - 1))

        # regularize
        coefficient = 1 - self.learning_rate * self.regularization_term
        self.weights_2 *= coefficient
        self.weights_1 *= coefficient

        # update weights
        self.weights_2 += self.learning_rate * np.dot(layer_2.transpose(), l2_delta)
        self.weights_1 += self.learning_rate * np.dot(layer_1.transpose(), l1_delta)

