import pyedflib
import plotly.express as px
from pathlib import Path
import os
import numpy as np


def read_files_from_dir(directory: Path):
    extensions = ["edf", "bdf"]

    return [(Path(file.path), pyedflib.highlevel.read_edf(file.path)) for file in os.scandir(directory) 
            if file.is_file() and file.name.endswith(tuple(extensions))]

def save_filtered_file(file: Path, filtered_signals: np.ndarray, output_path: str="data/filtered/"):
    _, signal_headers, header = pyedflib.highlevel.read_edf(file.absolute().as_posix())
    
    return pyedflib.highlevel.write_edf(output_path + file.stem, filtered_signals,
                                        signal_headers, header, digital=False)