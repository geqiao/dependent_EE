from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import pytest
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


def test_ind_sample_quasi_ot(data):
    input = data
    p = 4
    r = 100
    M = 200
    k = len(input)

    with NamedTemporaryFile() as tmp:
        file_path = tmp.name
        input.to_csv(file_path, index=False)
        result = ind_sample_from_quasi_OT(p, r, M, input_file_path=file_path)
    assert result.shape[1] == k
    assert result.shape[0] == r * (k + 1)
    assert len(result.iloc[:, 0].unique()) == p


def test_ind_sample_radical(data):
    r = 4
    input = data
    k = len(input)
    with NamedTemporaryFile() as tmp:
        file_path = tmp.name
        input.to_csv(file_path, index=False)
        result = ind_sample_from_radical_design(r, file_path)
    assert result.shape[1] == k
    assert result.shape[0] == r * (k + 1)
