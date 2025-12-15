# Not sure if we need functions that just reads and writes, but here they are. 
# Added more complex read and writes!

from pathlib import Path
import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)



def read_any(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def write_any(df: pd.DataFrame, path: Path):
    '''
    Writes a DataFrame to disk as CSV or Parquet.
    If extension missing/unsupported, defaults to .parquet.
    Returns the final path written.
    '''

    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()

    if ext == ".csv":
        df.to_csv(path, index=False)
        return path
    else:
        # default to parquet
        if ext not in {".parquet", ".pq"}:
            path = path.with_suffix(".parquet")    
        df.to_parquet(path, index=False)