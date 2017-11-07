from multiprocessing.pool import Pool

from neural_network import MultiLayerPerceptron
from presenter import save_plot
import numpy as np


def get_range(param):
    def _get_range(start, stop, step):
        return np.append(np.arange(start, stop, step), stop)

    return _get_range(param['min'], param['max'], param['step'])


def validate(params):
    classifier = params['classifier']
    data_set = params['data_set']
    left_index, right_index = params['indices_range']

    train_set = data_set[:int(left_index)] + data_set[int(right_index):]
    test_set = data_set[int(left_index):int(right_index)]

    classifier.init_weights()
    _epochs = classifier.learn(train_set=train_set)
    _accuracy = classifier.validate(data_set=test_set)

    k = params["index"]
    print(f'Cross validation k = {k} with accuracy {_accuracy} and {_epochs} epochs')
    return _accuracy, _epochs


def cross_validate(params, data_set):
    def init_case(indices_range, k):
        classifier = MultiLayerPerceptron(
            features=params['dataSize'],
            classes=params['classes'],
            hidden_neurons=params['hiddenNeurons']['value'],
            act_func=params['actFunc'],
            epochs=params['epochs'],
            min_mse=params['minMSE'],
            momentum=params['momentum']['value'],
            learning_rate=params['alpha']['value'],
            weights_deviation=params['weightsDeviation']['value']
        )
        return {
            'classifier': classifier,
            'data_set': data_set,
            'indices_range': indices_range,
            'index': k
        }

    k = params['validations']['value']
    test_size = len(data_set) / k
    indices_ranges = [(i * test_size, (i + 1) * test_size) for i in range(k)]

    cases = []
    for index, indices_range in enumerate(indices_ranges):
        cases.append(init_case(indices_range, index))

    with Pool(3) as pool:
        results = pool.map(validate, cases)

    _accuracies = list(map(lambda r: r[0], results))
    _epochs = list(map(lambda r: r[1], results))

    mean_accuracy = np.array(_accuracies).mean()
    mean_epochs = np.array(_epochs).mean()
    print(f'Learned network with mean: accuracy = {mean_accuracy}, epochs = {mean_epochs}')
    return mean_accuracy, mean_epochs


def research_parameter(params, data_set, param_name, param_long_name):
    def save_results():
        with open(f'./researches/{param_name}.txt', 'a') as param_results:
            param_results.write(f'Parameter {param_name}:\n')
            param_results.write(str(params) + '\n')
            param_results.write(f'Range: {param_range}\n')
            param_results.write(f'Accuracies = {accuracies}\n')
            param_results.write(f'Epochs = {epochs}\n\n')

        save_plot(param_range, accuracies, param_long_name, 'dokładność [%]', f'{param_name}')
        save_plot(param_range, epochs, param_long_name, 'liczba epok', f'{param_name}_epochs')

    param_range = get_range(params[param_name])
    accuracies = []
    epochs = []
    for param in param_range:
        print(f'Learning MLP with {param_name} = {param}')
        params_copy = params.copy()
        params_copy[param_name]['value'] = param
        acc, e = cross_validate(params_copy, data_set)
        accuracies.append(acc * 100)
        epochs.append(e)
    save_results()


def research(params, data_set):
    research_parameter(params, data_set, 'alpha', 'współczynnik uczenia')
    research_parameter(params, data_set, 'weightsDeviation', 'odchylenie wag początkowych')
    research_parameter(params, data_set, 'hiddenNeurons', 'liczba neuronów w warstwie ukrytej')
    research_parameter(params, data_set, 'momentum', 'momentum')
    research_parameter(params, data_set, 'validations', 'krotność walidacji krzyżowej')


def cross_research(params, data_set):
    def save_results(p, accuracy, epochs):
        with open(f'./researches/CROSS_RESEARCH.txt', 'a') as results:
            results.write(f'Accuracy = {accuracy}\n')
            results.write(f'Epochs = {epochs}\n')
            results.write(f'Parameters:\n')
            results.write(f'Alpha = {p["alpha"]}\n')
            results.write(f'Hidden neurons = {p["hiddenNeurons"]}\n')
            results.write(f'Momentum = {p["momentum"]}\n')
            results.write(f'Weights deviation = {p["weightsDeviation"]}\n')
            results.write(f'Validations = {p["validations"]}\n\n')

    max_accuracy = 0
    best_params = {}
    for p in params:
        mean_accuracy, mean_epochs = cross_validate(p, data_set)
        save_results(p, mean_accuracy, mean_epochs)
        if mean_accuracy > max_accuracy:
            print(f'NEW MAX VALUE! {mean_accuracy}')
            max_accuracy = mean_accuracy
            best_params = p.copy()
        print('Value less than last...')

    with open(f'./researches/CROSS_RESEARCH.txt', 'a') as results:
        results.write(f'Best result:\n')
        results.write(f'Accuracy = {max_accuracy}\n')
        results.write(f'Params:\n')
        results.write(best_params)
