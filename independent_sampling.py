import itertools

import numpy as np
import pandas as pd
import scipy.spatial.distance as sdis
import scipy.stats.qmc as qmc


def generate_combinations(V: np.ndarray, k: int) -> np.ndarray:
    """
    Generate all combinations contain k elements from a
    series of N elements
    :param V: a 1D vector with N elements
    :param k: number of elements contained in each sub vector
    :return: a 2D array in which each row is a k-element sub vector of V
    """
    return np.array(tuple(itertools.combinations(V, k)))


def ind_sample_from_quasi_OT(
    p: int, r: int, M: int, input_file_path: str
) -> pd.DataFrame:
    """
    Independent samples based on quasi-optimized trajectory design.
    A trajectory design is something like:
        x1,         x2,         x3
        a           b           c
        a+delta     b           c
        a+delta     b+delta     c
        a+delta     b+delta     c-delta
    :param p: number of levels from the min to max of the input.
              e.g. p= 4, the input range is divided as 0, 1/3,
              2/3 and 1
    :param r: number of quasi-optimized trajectories
    :param M: number of randomly generated trajectories in the input space
    :param input_file_path: path to the input csv file. The file should have
                            four columns: Nr., Parameter, Min, and Max. e.g:
                            Nr.      Parameter        Min           Max
                            1        x1               0             1
                            2        x2               2             3
                            ...

    :return: a dataframe contains r quasi-OT, the shape is (k+1) * r
             where k is the number of parameters
    """
    data = pd.read_csv(input_file_path, index_col=0, header=0)
    parameter_names = data["Parameter"]
    k = len(data)
    input_range = (data["Max"] - data["Min"]).values
    B = np.append(np.zeros([1, k]), np.tril(np.ones([k, k])), axis=0)
    delta = p / (2 * (p - 1))
    J_1 = np.ones([k + 1, k])
    J_k = J_1.copy()
    T = []
    A = np.eye(k)

    while len(T) < M:
        x_Star = np.random.randint(0, p - 1, [1, k]) / (p - 1)
        d = np.ones([1, k])
        d[np.random.rand(1, k) <= 0.5] = -1
        D_Star = np.eye(k)
        D_Star[np.diag_indices(k)] = d
        P_Star = A[np.random.permutation(k), :]
        B_Star = (J_1 * x_Star + (delta / 2) * ((2 * B - J_k).dot(D_Star) + J_k)).dot(
            P_Star
        )

        if k > 10:
            # when k is large (e.g. > 100), it is very likely the randomly generated
            # trajectory contains invalid point (i.e. has value outside the range [0,1])
            # and hence the trajectory will be discarded. The following code is used
            # to fine tune the invalid trajectory into valid one so to reduce the
            # time to generate valid random trajectories

            for cp in np.nditer(np.arange(p / 2)):
                B_Star[B_Star == (cp - 0.5 * p) / (p - 1)] = (cp + 0.5 * p) / (p - 1)
            for cp in np.nditer(np.arange(p / 2) + p / 2):
                B_Star[B_Star == (cp + 0.5 * p) / (p - 1)] = (cp - 0.5 * p) / (p - 1)

        if np.min(B_Star) >= 0 and np.max(B_Star) <= 1:
            T.append(B_Star)

    dist = np.zeros([M, M])
    for i in range(2, M + 1):
        for j in range(1, i):
            dist[i - 1, j - 1] = np.sum(sdis.cdist(T[i - 1], T[j - 1], "euclidean"))
            dist[j - 1, i - 1] = dist[i - 1, j - 1]

    vector = np.arange(1, M + 1)
    comb_dist = np.zeros([M, 1])
    comb_matrix = generate_combinations(vector, M - 1)
    discard_element = np.arange(M, 0, -1)

    for i in range(M, r, -1):
        if i == M:
            r_comb_matrix = generate_combinations(vector, 2)
            total_dist = 0.0
            for index in range(np.size(r_comb_matrix, 0)):
                total_dist = (
                    total_dist
                    + (dist[r_comb_matrix[index, 1] - 1, r_comb_matrix[index, 0] - 1])
                    ** 2
                )

            for j in range(i):
                comb_dist[j] = total_dist - np.sum(
                    dist[comb_matrix[j, :] - 1, discard_element[j] - 1] ** 2
                )
        else:
            for j in range(i):
                comb_dist[j] = comb_dist[j] - np.sum(
                    dist[comb_matrix[j, :] - 1, discard_element - 1] ** 2
                )

        index = np.argmax(comb_dist)
        old_vector = vector[:]
        vector = comb_matrix[index, :]
        discard_element = np.setdiff1d(old_vector, vector)
        comb_dist = np.delete(comb_dist, index)
        comb_matrix = np.delete(comb_matrix, index, 0)
        comb_matrix = comb_matrix[comb_matrix != discard_element]
        comb_matrix = np.reshape(comb_matrix, (i - 1, i - 2))

    quasi_optimized_combination = np.sort(vector)
    trajectories = [T[i] for i in quasi_optimized_combination - 1]

    input_ranges = np.tile(input_range, (k + 1, 1))
    input_mins = np.tile(data["Min"].values, (k + 1, 1))

    samples = [input_mins + trajectory * input_ranges for trajectory in trajectories]

    t = pd.DataFrame(np.concatenate(samples), columns=parameter_names)

    return t


def ind_sample_from_radical_design(r: int, input_file_path: str) -> pd.DataFrame:
    """
    Independent samples based on radical design. It utilizes the Sobol
    sequence to generate the random sample. A radical design is something
    like:
        x1,         x2,         x3
        a           b           c
        a+delta     b           c
        a           b+delta     c
        a           b           c-delta
    :param r: number of radical sets
    :param input_file_path: path to the input csv file. The file should have
                            four columns: Nr., Parameter, Min, and Max. e.g:
                            Nr.      Parameter        Min           Max
                            1        x1               0             1
                            2        x2               2             3
                            ...
    :return: a dataframe contains r quasi-OT, the shape is (k+1) * r
             where k is the number of parameters
    """
    data = pd.read_csv(input_file_path, index_col=0, header=0)
    parameter_names = data["Parameter"]
    k = len(data)
    sobol_seq = qmc.Sobol(4 * k, scramble=False).random(r + 5)[1:, :]

    def radical_design(input_mat: np.ndarray) -> np.ndarray:
        rows, cols = input_mat.shape
        output_mat = np.array([])
        for i in range(rows - 4):
            first_row = input_mat[i, : input_mat.shape[1] // 2]
            temp_mat = np.tile(first_row, [1 + cols // 2, 1])
            for j in range(1, cols // 2 + 1):
                temp_mat[j, j - 1] = input_mat[i + 4, cols // 2 + j - 1]
            if len(output_mat) == 0:
                output_mat = temp_mat
            else:
                output_mat = np.concatenate((output_mat, temp_mat))
        return output_mat

    samples = radical_design(
        np.concatenate((sobol_seq[:, :k], sobol_seq[:, -k:]), axis=1)
    )
    t = pd.DataFrame(samples, columns=parameter_names)
    return t
