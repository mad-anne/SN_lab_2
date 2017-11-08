import numpy as np
import os
from PIL import Image

CLASSES = 10


def _get_output_pattern(label):
    output = np.zeros(CLASSES)
    output[label] = 1
    return np.reshape(output, (1, CLASSES))


class Data:
    def __init__(self, data, filename, label=None):
        self.data = np.reshape(np.array(data), (1, len(data)))
        if label:
            self.label = int(label)
            self.output = _get_output_pattern(self.label)
        self.filename = filename


def _get_data_set_filenames(directory, ext):
    return [filename for filename in os.listdir(directory) if filename.endswith(ext)]


def _get_image_data(directory, filename, check_label=False):
    im = Image.open(directory + filename, 'r').convert('1')
    data = [int(d == 255) for d in list(im.getdata())]
    return Data(data, filename, filename[0]) if check_label else Data(data, filename)


def read_data_set(directory, ext):
    return [
        _get_image_data(directory, fn)
        for fn in _get_data_set_filenames(directory, ext)
    ]


def read_validate_set(directory, ext):
    return [
        _get_image_data(directory, fn, check_label=False)
        for fn in _get_data_set_filenames(directory, ext)
    ]
