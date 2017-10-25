class Matrix:
    def __init__(self, cols, rows):
        self.matrix = [[0 for x in range(rows)] for y in range(cols)]

    def set_value(self, col, row, value):
        self.matrix[col][row] = value
