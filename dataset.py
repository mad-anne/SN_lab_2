from multiprocessing.pool import Pool

import numpy as np
import os
from PIL import Image

CLASSES = 10


class Data:
    def __init__(self, data, path, label):
        self.data = np.reshape(np.array(data), (1, len(data)))
        self.label = int(label)
        self.output = np.reshape(np.array(data), (1, len(data)))
        self.filename = path


def _get_data_set_paths(directory):
    return (
        directory + folder + '/' + path
        for folder in os.listdir(directory)
        for path in os.listdir(directory + folder)
    )


def _get_image_data(path):
    im = Image.open(path, 'r').convert('1')
    data = [int(d == 255) for d in list(im.getdata())]
    return Data(data, path, path.split('/')[-2])


def read_data_set(directory):
    paths = _get_data_set_paths(directory)
    with Pool(4) as pool:
        results = pool.map(_get_image_data, paths)
    return results
