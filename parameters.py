import json


def read_parameters(filename):
    with open(filename, 'r') as f:
        return json.loads(f.read())
