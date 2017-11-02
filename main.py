from dataset import read_data_set
from neural_network import MultiLayerPerceptron

from activation_function import SigmoidFunction


dir = './dataset/'
ext = '.png'
data_set = read_data_set(dir, ext)

mlp = MultiLayerPerceptron(70, 10, 30, 0.01, SigmoidFunction())

mlp.learn(data_set, epochs=1000)

for data in data_set:
    print(f'Predicted value for {data.label} is {mlp.predict(data.data)}')
