import random
import numpy as np

clifford_rotations = ['I', 'X', 'Y', 'X/2', '-X/2', 'Y/2', '-Y/2', 'Y_X',
                      'X/2_Y/2', 'X/2_-Y/2', '-X/2_Y/2', '-X/2_-Y/2',
                      'Y/2_X/2', 'Y/2_-X/2', '-Y/2_X/2', '-Y/2_-X/2',
                      'X_Y/2', 'X_-Y/2', 'Y_X/2', 'Y_-X/2', '-X/2_Y/2_X/2',
                      '-X/2_-Y/2_X/2', 'X/2_Y/2_X/2', '-X/2_Y/2_-X/2']

gate_mat_dict = {
    'I': np.matrix([[1, 0], [0, 1]]),
    'X': np.matrix([[0, -1j], [-1j, 0]]),
    'X/2': np.matrix([[1, -1j], [-1j, 1]]) / np.sqrt(2),
    '-X/2': np.matrix([[1, 1j], [1j, 1]]) / np.sqrt(2),
    'Y': np.matrix([[0, -1], [1, 0]]),
    'Y/2': np.matrix([[1, -1], [1, 1]]) / np.sqrt(2),
    '-Y/2': np.matrix([[1, 1], [-1, 1]]) / np.sqrt(2)
}


def choose_random_clifford_gate():
    i = random.randint(0, len(clifford_rotations) - 1)
    return clifford_rotations[i]


def make_random_gates_string(length):
    gate_list = []
    for i in range(length):
        gate_list.append(choose_random_clifford_gate())
    if gate_list:
        return '_'.join(gate_list)
    else:
        return 'I'


def gates_string_to_mat(gates_string):
    current_mat = np.matrix([[1, 0], [0, 1]])
    for gate in gates_string.split('_'):
        gate_mat = gate_mat_dict[gate]
        current_mat = gate_mat * current_mat
    return current_mat


def mat_to_clifford_rotation(mat):
    for clif_rot in clifford_rotations:
        clif_mat = gates_string_to_mat(clif_rot)
        clif_mat_i = 1j * clif_mat
        if np.allclose(mat, clif_mat):
            return clif_rot
        elif np.allclose(mat, clif_mat_i):
            return clif_rot
        elif np.allclose(mat, -1 * clif_mat):
            return clif_rot
        elif np.allclose(mat, -1 * clif_mat_i):
            return clif_rot
    raise Exception('Could not find inversion for mat {}'.format(mat))


def decompose_mat_into_gates(mat):
    '''
    Unitary ((a, b), (c, d)) decomposed into:
        - angle of rotation theta
        - global phase alpha
        - axis of rotation defined by nx, ny and nz
    '''
    a, b, c, d = mat.flatten()
    a0 = (a + d) / 2
    a1 = (b + c) / 2
    a2 = (c - b) / (2 * 1j)
    a3 = (a - d) / 2
    theta = 2 * np.arccos(np.absolte(a0))
    alpha = np.phase(a0 / np.cos(theta / 2))
    nx = a1 / (-1j * np.exp(1j * alpha) * np.sin(theta / 2))
    ny = a2 / (-1j * np.exp(1j * alpha) * np.sin(theta / 2))
    nz = a3 / (-1j * np.exp(1j * alpha) * np.sin(theta / 2))
    return theta, alpha, nx, ny, nz
