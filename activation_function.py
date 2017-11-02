import numpy as np

from abc import ABC, abstractmethod


class ActivationFunction(ABC):
    @abstractmethod
    def get_output(self, x):
        pass


class SigmoidFunction(ActivationFunction):
    def get_output(self, x):
        return 1 / (1 + np.exp(-x))
