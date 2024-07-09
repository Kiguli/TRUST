import numpy as np
import pandas as pd
from numpy import ndarray


def read_file(file_path: str) -> ndarray | None:
    """Reads a file, determines its type (csv,json,txt) and returns a numpy array. Extension txt is expected to be
    comma delimited"""
    df = None
    # split the file path to get the file type
    file_type = file_path.split(".")[-1]
    try:
        if file_type == "csv":
            df = pd.read_csv(file_path)
        elif file_type == "json":
            df = pd.read_json(file_path)
        elif file_type == "txt":
            df = np.loadtxt(file_path, delimiter=',')
            return df
    except Exception:
        raise ValueError(f"Unsupported file type: {file_type}, or path does not exist: {file_path}")
    return df.to_numpy()


if __name__ == "__main__":
    # Replace 'file_path' with your actual file path
    csv_array = read_file('data.csv')
    if csv_array is not None:
        print("CSV to NumPy array:\n", csv_array)

    json_array = read_file('data.json')
    if json_array is not None:
        print("JSON to NumPy array:\n", json_array)

    txt_array = read_file('data.txt')
    if txt_array is not None:
        print("Text to NumPy array:\n", txt_array)
