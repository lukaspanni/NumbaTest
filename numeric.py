import numpy


def split_lr(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        return
    l_matrix = numpy.zeros(matrix.shape)
    r_matrix = matrix.astype(float)
    r, c = 0, 0
    l_matrix[r, c] = 1
    for i in range(r_matrix.shape[0] - 1):
        next_r = r + 1
        l_matrix[next_r, c] = r_matrix[next_r, c] / r_matrix[r, c]
        for j in range(r_matrix.shape[1]):
            r_matrix[next_r, j] = r_matrix[next_r, j] - (r_matrix[r, j] * l_matrix[next_r, c])
        r = next_r
        c += 1
        l_matrix[r, c] = 1
    return l_matrix, r_matrix


def solve_lr(l_matrix, r_matrix, b_vector):
    if l_matrix.shape != r_matrix.shape or b_vector.shape[0] != l_matrix.shape[0] or b_vector.shape[1] != 1:
        return
    y_vector = numpy.zeros(b_vector.shape)
    for i in range(l_matrix.shape[0]):
        tmp = b_vector[i]
        # TODO: implement


l_m, r_m = split_lr(numpy.array([[7, -4],
                                 [-2, 5]]))

print(l_m)
print(r_m)
