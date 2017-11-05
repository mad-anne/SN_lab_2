from activation_function import SigmoidFunction
from dataset import read_data_set
from parameters import read_parameters
from researches import research
import numpy as np

if __name__ == "__main__":
    params = read_parameters('parameters.json')
    params['actFunc'] = SigmoidFunction()

    data_set = read_data_set(params['dataSetDir'], params['dataSetExt'])
    np.random.shuffle(data_set)

    research(params, data_set)
