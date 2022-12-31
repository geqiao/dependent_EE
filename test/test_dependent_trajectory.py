from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import pytest
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


def test_dependent_sample_quasi_ot(data):
    input = data
    k = len(input)
    p = 4
    r = 100
    M = 200

    with NamedTemporaryFile() as input_file, NamedTemporaryFile() as ind_sample_file:
        input_file_path = input_file.name
        ind_sample_file_path = ind_sample_file.name
        input.to_csv(input_file_path, index=False)
        indep_sample = ind_sample_from_quasi_OT(p, r, M, input_file_path)
        indep_sample.to_csv(ind_sample_file_path, index=False)
        result = generate_dependent_sample(
            k=k,
            ind_sample_file_path=ind_sample_file_path,
            dist_type="uniform",
            corr_coef=np.array([0.1, 0.2, 0.3]),
            a=np.array([-np.pi, -np.pi, -np.pi]),
            b=np.array([np.pi, np.pi, np.pi]),
            design="quasi_ot",
        )

        assert result.shape[1] == k
        assert result.shape[0] == k * len(indep_sample)


def test_dependent_sample_radical(data):
    input = data
    k = len(input)
    r = 100
    with NamedTemporaryFile() as input_file, NamedTemporaryFile() as ind_sample_file:
        input_file_path = input_file.name
        ind_sample_file_path = ind_sample_file.name
        input.to_csv(input_file_path, index=False)
        indep_sample = ind_sample_from_radical_design(r, input_file_path)
        indep_sample.to_csv(ind_sample_file_path, index=False)

        result = generate_dependent_sample(
            k=k,
            ind_sample_file_path=ind_sample_file_path,
            dist_type="normal",
            corr_coef=np.array([0.9, 0.4, 0.01]),
            a=np.array([0, 0, 0]),
            b=np.array([1, 1, 1]),
            design="radical",
        )
        assert result.shape[1] == k
        assert result.shape[0] == k * len(indep_sample)
