import numpy as np
import pandas as pd
import scipy as sp


def original_dist(z, type, a, b):
    z_origin = z
    if type == "normal":
        if np.prod(a) == 0 and np.prod(b) == 0:
            z_origin = z
        else:
            z_cdf = sp.stats.norm.cdf(z)
            z_cdf[z_cdf == 1] = 1 - 1e-6
            z_cdf[z_cdf == 0] = 1e-6
            z_origin = sp.stats.norm.ppf(z_cdf)
    elif type == "uniform":
        z_origin = a + (b - a) * sp.stats.norm.cdf(z)
    return z_origin


def generate_dependent_sample(
    k: int,
    data_file: str,
    corrcoef: np.ndarray,
    type: str,
    a: np.ndarray,
    b: np.ndarray,
    design: str = "quasi_ot",
) -> pd.DataFrame:
    data = pd.read_csv(data_file, header=0)
    col_names = data.columns
    r = len(data) // (k + 1)
    data = data.replace(0, 1e-6)
    data = data.replace(1, 1 - 1e-6)

    # construct the correlation matrix
    C = np.triu(np.ones(k))
    C[(C - np.eye(k)) == 1] = corrcoef
    C[np.tril_indices_from(C, k=-1)] = corrcoef

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
    Out_Z_cor = []
    if design == "quasi_ot":
        index = np.ones((2 * k + 2), dtype=int)
        index[1::2] = np.arange(k + 1)
        index[0::2] = np.arange(k + 1)
        index = index[1:-1]
    elif design == "radical":
        index = np.ones(2 * k, dtype=int)
        index[0::2] = 0
        index[1::2] = index[1::2].cumsum()

    for temp_Z_indep in Z_indep.reshape(-1, k + 1, k):
        temp_Z_indep = temp_Z_indep[index, :]
        out_temp_Z_dep = []
        out_temp_Z_dep_total = []
        for j in range(k):

            cc = (
                np.argwhere(temp_Z_indep[2 * j + 1, :] - temp_Z_indep[2 * j, :])[0][0]
                if design == "quasi_ot"
                else j
            )
            if cc == k - 1:
                if len(out_temp_Z_dep) == 0:
                    out_temp_Z_dep = temp_Z_indep[2 * j : 2 * j + 2, :].dot(U[0])
                else:
                    out_temp_Z_dep = np.concatenate(
                        (out_temp_Z_dep, temp_Z_indep[2 * j : 2 * j + 2, :].dot(U[0]))
                    )
            else:
                shuffledtemp = np.roll(
                    temp_Z_indep[2 * j : 2 * j + 2, :], k - 1 - cc, axis=1
                ).dot(U[cc + 1])
                if len(out_temp_Z_dep) == 0:
                    out_temp_Z_dep = np.roll(shuffledtemp, -k + 1 + cc, axis=1)
                else:
                    out_temp_Z_dep = np.concatenate(
                        (out_temp_Z_dep, np.roll(shuffledtemp, -k + 1 + cc, axis=1))
                    )
            shuffledtemp = np.roll(temp_Z_indep[2 * j : 2 * j + 2, :], -cc, axis=1).dot(
                U[cc]
            )
            if len(out_temp_Z_dep_total) == 0:
                out_temp_Z_dep_total = np.roll(shuffledtemp, cc, axis=1)
            else:
                out_temp_Z_dep_total = np.concatenate(
                    (out_temp_Z_dep_total, np.roll(shuffledtemp, cc, axis=1))
                )
        if len(Out_Z_cor) == 0:
            Out_Z_cor = np.concatenate((out_temp_Z_dep, out_temp_Z_dep_total))
        else:
            Out_Z_cor = np.concatenate(
                (Out_Z_cor, out_temp_Z_dep, out_temp_Z_dep_total)
            )
    Out_Z_cor_ori = original_dist(Out_Z_cor, type, a, b)
    df_out = pd.DataFrame(Out_Z_cor_ori, columns=col_names)
    return df_out
