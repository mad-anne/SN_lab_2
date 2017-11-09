import numpy as np


class DimensionException(Exception):
    pass


def transpose(matrix):
    rows, cols = matrix.shape
    transposed_array = [matrix.item((row, col)) for col in range(cols) for row in range(rows)]
    return np.reshape(np.array(transposed_array), (cols, rows))


def dot(matrix_a, matrix_b):
    rows_a, cols_a = matrix_a.shape
    rows_b, cols_b = matrix_b.shape

    if cols_a != rows_b:
        raise DimensionException(
            f'Array with shape ({rows_a}, {cols_a}) and ({rows_b}, {cols_b}) do not match dimensions!')

    result = []
    for row_a in range(rows_a):
        for col_b in range(cols_b):
            slice_a = matrix_a[row_a, :]
            slice_b = matrix_b[:, col_b]
            result.append(sum([slice_a[i] * slice_b[i] for i in range(len(slice_a))]))
    return np.reshape(np.array(result), (rows_a, cols_b))
