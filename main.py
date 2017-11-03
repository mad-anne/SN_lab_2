from activation_function import SigmoidFunction
from dataset import read_data_set
from neural_network import MultiLayerPerceptron
from parameters import read_parameters
import numpy as np

params = read_parameters('parameters.json')


def _get_range(start, stop, step):
    return np.append(np.arange(start, stop, step), stop)


min_mse = params['minMSE']
epochs = params['epochs']
validations = params['validations']
data_size = params['dataSize']
classes = params['classes']
alpha = params['alpha']['value']
weights_deviation = params['weightsDeviation']['value']
hidden_neurons = params['hiddenNeurons']['value']

alpha_range = _get_range(
    params['alpha']['min'],
    params['alpha']['max'],
    params['alpha']['step']
)
weights_deviation_range = _get_range(
    params['weightsDeviation']['min'],
    params['weightsDeviation']['max'],
    params['weightsDeviation']['step']
)
hidden_neurons_range = _get_range(
    params['hiddenNeurons']['min'],
    params['hiddenNeurons']['max'],
    params['hiddenNeurons']['step']
)

dir = './dataset/'
ext = '.png'
data_set = read_data_set(dir, ext)

mlp = MultiLayerPerceptron(data_size, classes, hidden_neurons, alpha, SigmoidFunction())

mlp.learn(data_set, epochs=epochs)

for data in data_set:
    print(f'Predicted value for {data.label} is {mlp.predict(data.data)}')
