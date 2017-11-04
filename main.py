from activation_function import SigmoidFunction
from dataset import read_data_set
from parameters import read_parameters
from researches import research
import numpy as np

params = read_parameters('parameters.json')

data_set = read_data_set(params["dataSetDir"], params["dataSetExt"])
np.random.shuffle(data_set)

research(data_set, params, SigmoidFunction())
