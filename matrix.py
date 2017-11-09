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
            f'Arrays with shape ({rows_a}, {cols_a}) and ({rows_b}, {cols_b}) do not match dimensions!')

    result = [
        sum([matrix_a[row_a, i] * matrix_b[i, col_b] for i in range(cols_a)])
        for row_a in range(rows_a) for col_b in range(cols_b)]
    return np.reshape(np.array(result), (rows_a, cols_b))


def multiply(matrix, coefficient):
    rows, cols = matrix.shape
    result = [matrix.item((row, col)) * coefficient for row in range(rows) for col in range(cols)]
    return np.reshape(np.array(result), (rows, cols))


def add(matrix_a, matrix_b):
    rows_a, cols_a = matrix_a.shape
    rows_b, cols_b = matrix_b.shape

    if rows_a != rows_b or cols_a != cols_b:
        raise DimensionException(
            f'Arrays with shape ({rows_a}, {cols_a}) and ({rows_b}, {cols_b}) do not match dimensions!')

    result = [matrix_a[row, col] + matrix_b[row, col] for row in range(rows_a) for col in range(cols_a)]
    return np.reshape(np.array(result), (rows_a, cols_a))
