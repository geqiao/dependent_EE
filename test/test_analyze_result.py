from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import pytest
from analyze_ee_cor_result import agg_ee_results, compute_ee_cor
from dependent_sampling import generate_dependent_sample
from independent_sampling import (
    ind_sample_from_quasi_OT,
    ind_sample_from_radical_design,
)


@pytest.fixture
def data():
    df = pd.DataFrame(
        {
            "Nr.": np.arange(1, 4),
            "Parameter": [f"x{i}" for i in np.arange(1, 4)],
            "Min": [0] * 3,
            "Max": [1] * 3,
        }
    )
    return df


def model(df):
    return pd.DataFrame({"y1": df.x1 + df.x2 + df.x3, "y2": df.x1 * df.x2 * df.x3})


def test_analyze_result_quasi_ot(data):
    input = data
    k = len(input)
    p = 4
    r = 100
    M = 200
    design = "quasi_ot"

    with NamedTemporaryFile() as input_file, NamedTemporaryFile() as ind_sample_file, NamedTemporaryFile() as model_output_file:
        input_file_path = input_file.name
        ind_sample_file_path = ind_sample_file.name
        model_output_file_path = model_output_file.name
        input.to_csv(input_file_path, index=False)
        ind_sample = ind_sample_from_quasi_OT(p, r, M, input_file_path)
        ind_sample.to_csv(ind_sample_file_path, index=False)
        dep_sample = generate_dependent_sample(
            k=k,
            ind_sample_file_path=ind_sample_file_path,
            dist_type="uniform",
            corr_coef=np.array([0.1, 0.2, 0.3]),
            a=np.array([-np.pi, -np.pi, -np.pi]),
            b=np.array([np.pi, np.pi, np.pi]),
            design=design,
        )
        model_output = model(dep_sample)
        model_output.to_csv(model_output_file, index=False)
        EE_ind, EE_full = compute_ee_cor(
            k, ind_sample_file_path, model_output_file_path, design
        )
        agg_ee_results(EE_ind, EE_full)
        assert EE_ind.shape[0] == r
        assert EE_ind.shape[1] == model_output.shape[1]
        assert EE_ind.shape[2] == k
        assert EE_full.shape[0] == r
        assert EE_full.shape[1] == model_output.shape[1]
        assert EE_full.shape[2] == k
        result = agg_ee_results(EE_ind, EE_full)
        assert len(result.keys()) == 6


def test_analyze_result_radical(data):
    input = data
    k = len(input)
    r = 100
    design = "radical"
    with NamedTemporaryFile() as input_file, NamedTemporaryFile() as ind_sample_file, NamedTemporaryFile() as model_output_file:
        input_file_path = input_file.name
        ind_sample_file_path = ind_sample_file.name
        model_output_file_path = model_output_file.name
        input.to_csv(input_file_path, index=False)
        ind_sample = ind_sample_from_radical_design(r, input_file_path)
        ind_sample.to_csv(ind_sample_file_path, index=False)

        dep_sample = generate_dependent_sample(
            k=k,
            ind_sample_file_path=ind_sample_file_path,
            dist_type="normal",
            corr_coef=np.array([0.9, 0.4, 0.01]),
            a=np.array([0, 0, 0]),
            b=np.array([1, 1, 1]),
            design=design,
        )
        model_output = model(dep_sample)
        model_output.to_csv(model_output_file, index=False)
        EE_ind, EE_full = compute_ee_cor(
            k, ind_sample_file_path, model_output_file_path, design
        )
        agg_ee_results(EE_ind, EE_full)
        assert EE_ind.shape[0] == r
        assert EE_ind.shape[1] == model_output.shape[1]
        assert EE_ind.shape[2] == k
        assert EE_full.shape[0] == r
        assert EE_full.shape[1] == model_output.shape[1]
        assert EE_full.shape[2] == k
        result = agg_ee_results(EE_ind, EE_full)
        assert len(result.keys()) == 6
