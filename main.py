from dataset import read_data_set
from neural_network import Layer


dir = './dataset/'
ext = '.png'
data_set = read_data_set(dir, ext)

layer = Layer(2, 4, None)
layer.init_random_weights(deviation=0.1)
