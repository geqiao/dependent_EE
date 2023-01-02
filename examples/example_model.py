import pandas as pd


def model(df: pd.DataFrame) -> pd.DataFrame:
    """
    this is a test model with three inputs x1, x2, x3
    and two outputs y1 and y2, where y1 = x1 + x2 + x3
    and y2 = x1 + x2 * x3
    """

    return pd.DataFrame({"y1": df.x1 + df.x2 + df.x3, "y2": df.x1 * df.x2 * df.x3})
