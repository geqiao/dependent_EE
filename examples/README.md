# Introduction

This folder contains examples of the usage of the code.

# Note

- To run the example code, use for instance:

  > python examples/quasi_OT_design/main.py

- The inputs should be stored in the `data` folder in `.csv` format.
  There are three csv files needed:

  - `unit_input.csv`: stores the unit range of the model inputs and will be used to generate the independent samples. The file should be looked like, e.g.:
    ```csv
    Nr.,    Parameter,  Min,    Max
    1,      x1,         0,      1
    2,      x2,         0,      1
    3,      x3,         0,      1
    ```
  - `dist.csv`: stores the original distribution of the inputs. `a` and `b` are
    parameters of the distribution (for details see the `original_dist` function in `src/dependent_sampling.py`). The format should look like:
    ```csv
    Parameter,  dist_type,  a,  b
    x1,         uniform,    0,  10
    x2,         uniform,    0,  20
    x3,         normal,     10, 3
    ```
  - `correlation_matrix.csv`: stores the correlation matrix, e.g.:
    ```csv
    Parameter,    x1,     x2,     x3
    x1,           1,      0.1,    0.2
    x2,           0.1,    1,      0.3
    x3,           0.2,    0.3,    1
    ```

- The current implementation of the function `original_dist` in `src/dependent_sampling.py` only deals with _uniform_ and _Gaussian_ distribution of the model inputs. If the user's model contain inputs from other distributions, an inverse transformation function is hence needed to transform standard normal distributed variable `Z` into the target distribution.

- An example model is provided in `example_model.py` which has 3 inputs and 2 outputs for demonstration purpose. The user should either implement their own model and import in the `main.py`, or use the generated `dep_sample.csv` as model inputs and save the outputs in `model_output.csv` with the following format:

  ```csv
  y1,    y2,     ...
  123,   434,    ...
  ...
  ```

- The output of the analysis is stored in the file `ee_result.csv`
