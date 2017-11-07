from activation_function import SigmoidFunction
from dataset import read_data_set
from parameters import read_parameters
from researches import cross_research, research
import numpy as np


def get_crossed_params(params):
    crossed_params = []
    for alpha in params['crossParams']['alpha']:
        for neurons in params['crossParams']['hiddenNeurons']:
            for momentum in params['crossParams']['momentum']:
                for weightsDeviation in params['crossParams']['weightsDeviation']:
                    for validations in params['crossParams']['validations']:
                        params_copy = params.copy()
                        params_copy['alpha']['value'] = alpha
                        params_copy['hiddenNeurons']['value'] = neurons
                        params_copy['momentum']['value'] = momentum
                        params_copy['weightsDeviation']['value'] = weightsDeviation
                        params_copy['validations']['value'] = validations
                        crossed_params.append(params_copy)
    return crossed_params


if __name__ == "__main__":
    params = read_parameters('parameters.json')
    params['actFunc'] = SigmoidFunction()

    if params['research']:
        data_set = read_data_set(params['dataSetDir'], params['dataSetExt'])
        np.random.shuffle(data_set)
        research(params, data_set)

    if params['crossResearch']:
        data_set = read_data_set(params['dataSetDir'], params['dataSetExt'])
        np.random.shuffle(data_set)
        crossed_params = get_crossed_params(params)
        cross_research(crossed_params, data_set)

    if params['validate']:
        data_set = read_data_set(params['validateDir'], params['validateExt'])
