from typing import Dict

import numpy as np
import pandas as pd


def compute_ee_cor(
    k: int,
    ind_sample_file_path: str,
    model_result_file_path: str,
    design: str = "quasi_ot",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the EE based on trajectory design
    :param k: number of parameters
    :param ind_sample_file_path: file path of the generated independent samples
    :param model_result_file_path: file path of the model results. Number of columns equals
                                to the number of model outputs, while number of rows is the
                                same as the number of rows in the dependent samples
    :param design: str, either `quasi_ot` or `radical`
    :return: a tuple of independent EE and full EE
    """
    input_data = pd.read_csv(ind_sample_file_path, header=0).values
    result_data = pd.read_csv(model_result_file_path, header=0).values
    r = len(input_data) // (k + 1)
    dimension = result_data.shape[1]
    EE_ind = np.zeros((r, dimension, k))
    EE_full = np.zeros((r, dimension, k))

    for i in range(r):
        diff_res = (
            result_data[1 + k * i * 4 : k * 4 * (i + 1) : 2, :]
            - result_data[k * i * 4 : k * 4 * (i + 1) - 1 : 2, :]
        )
        if design == "quasi_ot":
            delta_matrix = (
                input_data[1 + (k + 1) * i : (k + 1) * (i + 1), :]
                - input_data[(k + 1) * i : (k + 1) * (i + 1) - 1, :]
            )
        elif design == "radical":
            delta_matrix = (
                input_data[1 + (k + 1) * i : (k + 1) * (i + 1), :]
                - input_data[(k + 1) * i, :]
            )
        else:
            raise NotImplementedError(f"The {design} design is not implemented yet.")

        delta_position_filter = delta_matrix != 0

        for j in range(k):
            if (
                delta_matrix[j, delta_position_filter[j, :]] == 0
                or len(delta_matrix[j, delta_position_filter[j, :]]) == 0
            ):
                EE_ind[i, :, delta_position_filter[j, :]] = 0
                EE_full[i, :, delta_position_filter[j, :]] = 0
            else:
                EE_ind[i, :, delta_position_filter[j, :]] = diff_res[j, :] / (
                    delta_matrix[j, delta_position_filter[j, :]]
                )
                EE_full[i, :, delta_position_filter[j, :]] = diff_res[j + k, :] / (
                    delta_matrix[j, delta_position_filter[j, :]]
                )
    return EE_ind, EE_full


def agg_ee_results(EE_ind: np.ndarray, EE_full: np.ndarray) -> Dict[str, float]:
    """
    Aggregate the results of elementary effects.
    :param EE_ind: 3 dimensional array of independent elementary effects with the
        shape [number of trajectories/radicals, number of model outputs, number
        of model inputs]
    :param EE_full: 3 dimensional array of full elementary effects with the
        shape [number of trajectories/radicals, number of model outputs, number
        of model inputs]
    :return: a dictionary of the aggregated results including the mean, abs_mean
        and standard deviation of the independent EE and full EE
    """

    mean_ind = np.mean(EE_ind, axis=0).T
    abs_mean_ind = np.mean(np.abs(EE_ind), axis=0).T
    std_ind = np.std(EE_ind, axis=0).T

    norm_mean_full = np.mean(EE_full, axis=0).T
    abs_mean_full = np.mean(np.abs(EE_full), axis=0).T
    std_full = np.std(EE_full, axis=0).T

    return dict(
        mean_ind=mean_ind,
        abs_mean_ind=abs_mean_ind,
        std_ind=std_ind,
        mean_full=norm_mean_full,
        abs_mean_ful=abs_mean_full,
        std_full=std_full,
    )
