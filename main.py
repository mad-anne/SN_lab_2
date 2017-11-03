from activation_function import SigmoidFunction
from dataset import read_data_set
from neural_network import MultiLayerPerceptron
from parameters import read_parameters
import numpy as np


def get_range(param):
    def _get_range(start, stop, step):
        return np.append(np.arange(start, stop, step), stop)

    return _get_range(param["min"], param["max"], param["step"])


def cross_validate(classifier, data_set, k, weights_deviation, epochs, min_mse):
    test_size = len(data_set) / k
    np.random.shuffle(data_set)
    left_index = 0
    right_index = test_size
    _accuracies = []
    _epochs = []
    for k in range(k):
        train_set = data_set[:int(left_index)] + data_set[int(right_index):]
        test_set = data_set[int(left_index):int(right_index)]
        classifier.init_weights(weights_deviation)
        _epochs.append(classifier.learn(train_set=train_set, epochs=epochs, min_mse=min_mse))
        _accuracies.append(classifier.validate(data_set=test_set))
        print(f'Cross validation k = {k} with accuracy {_accuracies[-1]} and {_epochs[-1]} epochs')
        left_index = right_index
        right_index += test_size
    mean_accuracy = sum(_accuracies) / len(_accuracies)
    mean_epochs = sum(_epochs) / len(_epochs)
    return mean_accuracy, mean_epochs


params = read_parameters('parameters.json')

data_set = read_data_set(
    directory=params["dataSetDir"],
    ext=params["dataSetExt"]
)

mlp = MultiLayerPerceptron(
    features=params['dataSize'],
    classes=params['classes'],
    hidden_neurons=params['hiddenNeurons']['value'],
    learning_rate=params['alpha']['value'],
    act_func=SigmoidFunction()
)

print('Learning neural network...')

accuracy, epochs = cross_validate(
    classifier=mlp,
    data_set=data_set,
    k=params['validations'],
    weights_deviation=params['weightsDeviation']['value'],
    epochs=params['epochs'],
    min_mse=params['minMSE']
)

print(f'Accuracy is {accuracy*100}% with mean {epochs} epochs')

alpha_range = get_range(params['alpha'])
weights_deviation_range = get_range(params['weightsDeviation'])
hidden_neurons_range = get_range(params['hiddenNeurons'])
