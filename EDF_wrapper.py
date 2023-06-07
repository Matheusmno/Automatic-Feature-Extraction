import pyedflib
import plotly.express as px
from pathlib import Path
import os
import numpy as np


def read_files_from_dir(directory: Path):
    extensions = ["edf", "bdf"]

    return [Path(file) for file in os.scandir(directory) 
            if file.is_file() and file.name.endswith(tuple(extensions))]

def save_filtered_file(file: Path, filtered_signals: np.ndarray, output_path: str="data/filtered/"):
    original_file = pyedflib.EdfReader(file.absolute().as_posix())

    # Create a new EDF file to write the modified signals into
    new_file = pyedflib.EdfWriter(output_path + f'{file.stem}.bdf',
                                  n_channels=original_file.signals_in_file,
                                  file_type=pyedflib.FILETYPE_BDF)

    # Keeping both header and signal headers unmodified
    new_file.setHeader(original_file.getHeader())
    new_file.setSignalHeaders(original_file.getSignalHeaders())

    # Write the filtered signals to new file
    new_file.writeSamples(filtered_signals)

    new_file.close()
    original_file.close()