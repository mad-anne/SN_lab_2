import unittest
import numpy as np

from matrix import transpose, dot


class MatrixTestCase(unittest.TestCase):
    def setUp(self):
        self.matrix_a = np.reshape(np.array([1, 2, 3, 4]), (2, 2))
        self.matrix_b = np.reshape(np.array([1, 2, 3, 1, 2, 3]), (2, 3))

    def test_transpose(self):
        matrix_c = transpose(self.matrix_b)
        transposed_matrix = np.reshape(np.array([1, 1, 2, 2, 3, 3]), (3, 2))
        self.assertTrue(np.array_equal(matrix_c, transposed_matrix))

    def test_dot(self):
        matrix_c = dot(self.matrix_a, self.matrix_b)
        result_matrix = np.reshape(np.array([3, 6, 9, 7, 14, 21]), (2, 3))
        self.assertTrue(np.array_equal(matrix_c, result_matrix))
