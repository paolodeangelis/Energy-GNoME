import pandas as pd


def detect_outliers_iqr(df: pd.DataFrame, iqr_w: float = 1.5):
    """Detects outliers in a Pandas DataFrame using the IQR method.

    Args:
        df (DataFrame): Pandas DataFrame with numerical data.
        iqr_w (float): IQR width, default value 1.4

    Returns:
        _type_: A Pandas DataFrame of the same shape as `df` with True for outliers and False otherwise.
    """

    outlier_mask = pd.DataFrame(data=False, index=df.index, columns=df.columns)

    for col in df.columns:
        if (
            df[col].dtype.kind in "ifc"
        ):  # Check if the column type is numeric (boolean, integer, float, complex)
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask[col] = (df[col] < (Q1 - iqr_w * IQR)) | (df[col] > (Q3 + iqr_w * IQR))

    return outlier_mask
