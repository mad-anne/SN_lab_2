from dataset import read_data_set
from neural_network import MultiLayerPerceptron

from activation_function import SigmoidFunction


dir = './dataset/'
ext = '.png'
data_set = read_data_set(dir, ext)

mlp = MultiLayerPerceptron(70, 10, 20, 0.01, SigmoidFunction())

for data in data_set:
    print(f'Predicted value for {data.label} is {mlp.predict(data.data)}')
