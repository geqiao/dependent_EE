import itertools

import numpy as np
import pandas as pd
import scipy.spatial.distance as sdis


def n_choose_k(A, num):
    return np.array(tuple(itertools.combinations(A, num)))


def trajectory(p: int, r: int, M: int, parameter_range_csv_file: str) -> pd.DataFrame:
    data = pd.read_csv(parameter_range_csv_file, index_col=0, header=0)
    parameter_names = data["Parameter"]
    k = data.shape[0]
    datamatrix_diff = (data["Max"] - data["Min"]).values
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
    comb_matrix = n_choose_k(vector, M - 1)
    discard_element = np.arange(M, 0, -1)

    for i in range(M, r, -1):
        if i == M:
            r_comb_matrix = n_choose_k(vector, 2)
            CombDist_total = 0.0
            for index in range(np.size(r_comb_matrix, 0)):
                CombDist_total = (
                    CombDist_total
                    + (dist[r_comb_matrix[index, 1] - 1, r_comb_matrix[index, 0] - 1])
                    ** 2
                )

            for j in range(i):
                comb_dist[j] = CombDist_total - np.sum(
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

    best_comb = np.sort(vector)
    trajectory_set = [T[i] for i in best_comb - 1]

    datamatrix_diff_transver = np.tile(datamatrix_diff, (k + 1, 1))
    data_transver = np.tile(data["Min"].values, (k + 1, 1))

    parameter_set = [
        data_transver + trajectory * datamatrix_diff_transver
        for trajectory in trajectory_set
    ]

    t = pd.DataFrame(np.concatenate(parameter_set), columns=parameter_names)

    return t
