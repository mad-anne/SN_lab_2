import os
from PIL import Image


class Data:
    def __init__(self, data, label):
        self.data = data
        self.label = label


def get_data_set_filenames(dir, ext):
    return [filename for filename in os.listdir(dir) if filename.endswith(ext)]


def _get_image_data(dir, filename):
    im = Image.open(dir + filename, 'r').convert('1')
    data = [int(d == 255) for d in list(im.getdata())]
    label = filename[0]
    return Data(data, label)


def read_data_set(dir, ext):
    return [_get_image_data(dir, fn) for fn in get_data_set_filenames(dir, ext)]
