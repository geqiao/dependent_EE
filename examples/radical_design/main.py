import os

import numpy as np
import pandas as pd
from examples.example_model import model
from src.analyze_ee_cor_result import agg_ee_results, compute_ee_cor
from src.dependent_sampling import generate_dependent_sample
from src.independent_sampling import ind_sample_from_radical_design

# file paths
path = os.path.dirname(__file__)
input_file_path = os.path.join(path, "data/unit_input.csv")
corr_matrix_file_path = os.path.join(path, "data/correlation_matrix.csv")
dist_file_path = os.path.join(path, "data/dist.csv")
ind_sample_file_path = os.path.join(path, "data/ind_sample.csv")
dep_sample_file_path = os.path.join(path, "data/dep_sample.csv")
model_output_file_path = os.path.join(path, "data/model_output.csv")
ee_results_file_path = os.path.join(path, "data/ee_results.csv")

# config for quasi_OT
k = 3
r = 100
design = "radical"


def read_correlation_matrix(file_path: str) -> np.ndarray:
    corr_matrix = pd.read_csv(file_path, index_col=0)
    k = len(corr_matrix)
    corr_coef = corr_matrix.values[np.triu(np.ones(k), 1) > 0].flatten()
    return corr_coef


if __name__ == "__main__":

    # generate independent samples
    ind_sample = ind_sample_from_radical_design(r, input_file_path)
    ind_sample.to_csv(ind_sample_file_path, index=False)

    # generate dependent samples
    dist = pd.read_csv(dist_file_path, index_col=0)
    corr_coef = read_correlation_matrix(corr_matrix_file_path)
    dep_sample = generate_dependent_sample(
        k=k,
        ind_sample_file_path=ind_sample_file_path,
        dist_type=dist["dist_type"].values,
        corr_coef=corr_coef,
        a=dist["a"].values,
        b=dist["b"].values,
        design=design,
    )
    dep_sample.to_csv(dep_sample_file_path, index=False)

    # compute model outputs
    model_output = model(dep_sample)
    model_output.to_csv(model_output_file_path, index=False)

    # compute EE
    EE_ind, EE_full = compute_ee_cor(
        k, ind_sample_file_path, model_output_file_path, design
    )

    # aggregate the results
    model_output_names = model_output.columns
    model_input_names = dist.index.values
    index = pd.MultiIndex.from_product(
        [model_input_names, model_output_names], names=["model_inputs", "model_outputs"]
    )

    agg_results = agg_ee_results(EE_ind, EE_full)
    result_df = (
        pd.DataFrame(
            index=agg_results.keys(),
            columns=index,
            data=np.array([_ for _ in agg_results.values()]).reshape(6, -1),
        )
        .T.reorder_levels(order=[1, 0])
        .sort_index()
        .T
    )
    result_df.to_csv(ee_results_file_path)
