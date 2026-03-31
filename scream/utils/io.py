"""
I/O helpers for SCREAM.

Convention
----------
All tabular data written by SCREAM scripts (generated background samples,
predictions, results) is stored as CSV.  Raw input data may be FITS or CSV;
read_table() handles both transparently and always returns a pandas DataFrame.
"""

from pathlib import Path

import pandas as pd
from astropy.table import Table


def read_table(path: str | Path) -> pd.DataFrame:
    """Load a FITS or CSV file and return a pandas DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to a CSV or FITS file.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    # astropy handles FITS and other formats; convert to pandas
    return Table.read(path).to_pandas()


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    """Write a DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
    path : str or Path
        Destination path. Parent directory is created if it does not exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
