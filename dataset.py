import numpy as np
import os
from PIL import Image

CLASSES = 10


class Data:
    def __init__(self, data, label, filename):
        self.data = np.reshape(np.array(data), (1, len(data)))
        self.label = int(label)
        self.output = np.zeros(CLASSES)
        self.output[self.label] = 1
        self.filename = filename


def _get_data_set_filenames(dir, ext):
    return [filename for filename in os.listdir(dir) if filename.endswith(ext)]


def _get_image_data(dir, filename):
    im = Image.open(dir + filename, 'r').convert('1')
    data = [int(d == 255) for d in list(im.getdata())]
    label = filename[0]
    return Data(data, label, filename)


def read_data_set(dir, ext):
    return [_get_image_data(dir, fn) for fn in _get_data_set_filenames(dir, ext)]
