import numpy as np
import pandas as pd
import scipy as sp


def original_dist(
    vector: np.ndarray, dist_type: str, a: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """
    Restore a vector from uniform distribution to its original distribution
    :param vector: vector in uniform distribution
    :param dist_type: type of original distribution
    :param a: lower boundary of a uniform distribution (only needed when
                the original distribution is uniform)
    :param b: upper boundary of a uniform distribution (only needed when
                the original distribution is uniform)
    :return: a vector in original distribution
    """
    if dist_type == "normal":
        if np.prod(a) == 0 and np.prod(b) == 0:
            vector_ori = vector
        else:
            z_cdf = sp.stats.norm.cdf(vector)
            z_cdf[z_cdf == 1] = 1 - 1e-6
            z_cdf[z_cdf == 0] = 1e-6
            vector_ori = sp.stats.norm.ppf(z_cdf)
    elif dist_type == "uniform":
        vector_ori = a + (b - a) * sp.stats.norm.cdf(vector)
    else:
        raise NotImplementedError(f"Distribution {dist_type} is not implemented.")
    return vector_ori


def generate_dependent_sample(
    k: int,
    ind_sample_file_path: str,
    corr_coef: np.ndarray,
    dist_type: str,
    a: np.ndarray,
    b: np.ndarray,
    design: str = "quasi_ot",
) -> pd.DataFrame:
    """
    Function to produce dependent samples. It is based on the independent samples from
    either quasi-OT design, or radical design
    :param k: number of parameters
    :param ind_sample_file_path: file path of the generated independent samples
    :param corr_coef: a vector of correlation coefficients. For instance for a correlation
                    matrix of 3 variables, the vector is [corr(x1,x2), corr(x1,x3), corr(x2,x3)]
    :param dist_type: type of the original distribution
    :param a: lower boundary of a uniform distribution (only needed when
                the original distribution is uniform)
    :param b: upper boundary of a uniform distribution (only needed when
                the original distribution is uniform)
    :param design: design of the independent random sample generation. either `quasi_ot`
        or `radical`
    :return: a dataframe of dependent samples. Its shape is [k(k+1)r, k]
    """
    data = pd.read_csv(ind_sample_file_path, header=0)
    col_names = data.columns
    data = data.replace(0, 1e-6)
    data = data.replace(1, 1 - 1e-6)

    # construct the correlation matrix
    C = np.triu(np.ones(k))
    C[(C - np.eye(k)) == 1] = corr_coef
    C[np.tril_indices_from(C, k=-1)] = corr_coef

    U = []
    for i in range(k):
        if i > 0:
            C = np.concatenate(
                (
                    np.concatenate((C[1:, 1:], C[:1, 1:]), axis=0),
                    np.concatenate((C[:1, 1:].T, np.array([[1]])), axis=0),
                ),
                axis=1,
            )
        U.append(np.linalg.cholesky(C).T)

    Z_indep = sp.stats.norm.ppf(data.values)
    Z_dep = []
    if design == "quasi_ot":
        filter = np.ones((2 * k + 2), dtype=int)
        filter[1::2] = np.arange(k + 1)
        filter[0::2] = np.arange(k + 1)
        filter = filter[1:-1]
    elif design == "radical":
        filter = np.ones(2 * k, dtype=int)
        filter[0::2] = 0
        filter[1::2] = filter[1::2].cumsum()
    else:
        raise NotImplementedError(f"Design {design} is not implemented.")

    for Z in Z_indep.reshape(-1, k + 1, k):
        Z = Z[filter, :]
        Z_dep_ind = []  # correlated samples for computing ind EE
        Z_dep_full = []  # correlated samples for computing full EE
        for j in range(k):

            col_index = (
                np.argwhere(Z[2 * j + 1, :] - Z[2 * j, :])[0][0]
                if design == "quasi_ot"
                else j
            )
            if col_index == k - 1:
                if len(Z_dep_ind) == 0:
                    Z_dep_ind = Z[2 * j : 2 * j + 2, :].dot(U[0])
                else:
                    Z_dep_ind = np.concatenate(
                        (Z_dep_ind, Z[2 * j : 2 * j + 2, :].dot(U[0]))
                    )
            else:
                shuffled_Z = np.roll(
                    Z[2 * j : 2 * j + 2, :], k - 1 - col_index, axis=1
                ).dot(U[col_index + 1])
                if len(Z_dep_ind) == 0:
                    Z_dep_ind = np.roll(shuffled_Z, -k + 1 + col_index, axis=1)
                else:
                    Z_dep_ind = np.concatenate(
                        (Z_dep_ind, np.roll(shuffled_Z, -k + 1 + col_index, axis=1))
                    )
            shuffled_Z = np.roll(Z[2 * j : 2 * j + 2, :], -col_index, axis=1).dot(
                U[col_index]
            )
            if len(Z_dep_full) == 0:
                Z_dep_full = np.roll(shuffled_Z, col_index, axis=1)
            else:
                Z_dep_full = np.concatenate(
                    (Z_dep_full, np.roll(shuffled_Z, col_index, axis=1))
                )
        if len(Z_dep) == 0:
            Z_dep = np.concatenate((Z_dep_ind, Z_dep_full))
        else:
            Z_dep = np.concatenate((Z_dep, Z_dep_ind, Z_dep_full))
    samples = original_dist(Z_dep, dist_type, a, b)
    df_out = pd.DataFrame(samples, columns=col_names)
    return df_out
