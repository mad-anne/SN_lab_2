from neural_network import MultiLayerPerceptron
from presenter import save_plot
import numpy as np


def get_range(param):
    def _get_range(start, stop, step):
        return np.append(np.arange(start, stop, step), stop)

    return _get_range(param["min"], param["max"], param["step"])


def cross_validate(classifier, data_set, k, weights_deviation, epochs, min_mse, m, learning_rate):
    test_size = len(data_set) / k
    left_index = 0
    right_index = test_size
    _accuracies = []
    _epochs = []
    for k in range(k):
        train_set = data_set[:int(left_index)] + data_set[int(right_index):]
        test_set = data_set[int(left_index):int(right_index)]
        classifier.init_weights(weights_deviation)
        _epochs.append(
            classifier.learn(
                train_set=train_set,
                epochs=epochs,
                min_mse=min_mse,
                momentum=m,
                learning_rate=learning_rate
            )
        )
        _accuracies.append(classifier.validate(data_set=test_set))
        print(f'Cross validation k = {k} with accuracy {_accuracies[-1]} and {_epochs[-1]} epochs')
        left_index = right_index
        right_index += test_size
    mean_accuracy = sum(_accuracies) / len(_accuracies)
    mean_epochs = sum(_epochs) / len(_epochs)
    print(f'Learned network with mean: accuracy = {mean_accuracy}, epochs = {mean_epochs}')
    return mean_accuracy, mean_epochs


def research_alphas(data_set, params, act_func):
    alpha_range = get_range(params['alpha'])
    accuracies = []
    epochs = []
    mlp = MultiLayerPerceptron(
        features=params['dataSize'],
        classes=params['classes'],
        hidden_neurons=params['hiddenNeurons']['value'],
        act_func=act_func
    )
    for alpha in alpha_range:
        print(f'Learning MLP with alpha = {alpha}')
        acc, e = cross_validate(
            classifier=mlp,
            data_set=data_set,
            k=params['validations'],
            weights_deviation=params['weightsDeviation']['value'],
            epochs=params['epochs'],
            min_mse=params['minMSE'],
            m=params['momentum']['value'],
            learning_rate=alpha
        )
        accuracies.append(acc * 100)
        epochs.append(e)
    save_plot(alpha_range, accuracies, 'współczynnik uczenia', 'dokładność [%]', 'alpha')
    save_plot(alpha_range, epochs, 'współczynnik uczenia', 'liczba epok', 'alpha_epochs')


def research_weights(data_set, params, act_func):
    weights_range = get_range(params['weightsDeviation'])
    accuracies = []
    epochs = []
    mlp = MultiLayerPerceptron(
        features=params['dataSize'],
        classes=params['classes'],
        hidden_neurons=params['hiddenNeurons']['value'],
        act_func=act_func
    )
    for deviation in weights_range:
        print(f'Learning MLP with weights deviation = {deviation}')
        acc, e = cross_validate(
            classifier=mlp,
            data_set=data_set,
            k=params['validations'],
            weights_deviation=deviation,
            epochs=params['epochs'],
            min_mse=params['minMSE'],
            m=params['momentum']['value'],
            learning_rate=params['alpha']['value']
        )
        accuracies.append(acc * 100)
        epochs.append(e)
    save_plot(weights_range, accuracies, 'odchylenie wag początkowych', 'dokładność [%]', 'weights')
    save_plot(weights_range, epochs, 'odchylenie wag początkowych', 'liczba epok', 'weights_epochs')


def research_hidden_neurons(data_set, params, act_func):
    hidden_neurons_range = get_range(params['hiddenNeurons'])
    accuracies = []
    epochs = []
    for neurons in hidden_neurons_range:
        print(f'Learning MLP with hidden neurons = {neurons}')
        mlp = MultiLayerPerceptron(
            features=params['dataSize'],
            classes=params['classes'],
            hidden_neurons=neurons,
            act_func=act_func
        )
        acc, e = cross_validate(
            classifier=mlp,
            data_set=data_set,
            k=params['validations'],
            weights_deviation=params['weightsDeviation']['value'],
            epochs=params['epochs'],
            min_mse=params['minMSE'],
            m=params['momentum']['value'],
            learning_rate=params['alpha']['value']
        )
        accuracies.append(acc * 100)
        epochs.append(e)
    save_plot(hidden_neurons_range, accuracies, 'liczba neuronów w warstwie ukrytej', 'dokładność [%]', 'neurons')
    save_plot(hidden_neurons_range, epochs, 'liczba neuronów w warstwie ukrytej', 'liczba epok', 'neurons_epochs')


def research_momentum(data_set, params, act_func):
    momentum_range = get_range(params['momentum'])
    accuracies = []
    epochs = []
    mlp = MultiLayerPerceptron(
        features=params['dataSize'],
        classes=params['classes'],
        hidden_neurons=params['hiddenNeurons']['value'],
        act_func=act_func
    )
    for momentum in momentum_range:
        print(f'Learning MLP with momentum = {momentum}')
        acc, e = cross_validate(
            classifier=mlp,
            data_set=data_set,
            k=params['validations'],
            weights_deviation=params['weightsDeviation']['value'],
            epochs=params['epochs'],
            min_mse=params['minMSE'],
            m=momentum,
            learning_rate=params['alpha']['value']
        )
        accuracies.append(acc * 100)
        epochs.append(e)
    save_plot(momentum_range, accuracies, 'momentum', 'dokładność [%]', 'momentum')
    save_plot(momentum_range, epochs, 'momentum', 'liczba epok', 'momentum_epochs')


def research(data_set, params, act_func):
    research_alphas(data_set, params, act_func)
    research_weights(data_set, params, act_func)
    research_hidden_neurons(data_set, params, act_func)
    research_momentum(data_set, params, act_func)
