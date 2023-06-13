import pyedflib
import plotly.express as px
from pathlib import Path
import os
import numpy as np


def read_files_from_dir(directory: Path):
    extensions = ["edf", "bdf"]
    files = []

    for file in os.scandir(directory):
        if file.is_file() and file.name.endswith(tuple(extensions)):
            filepath = file.path
            signals, signal_headers, header = pyedflib.highlevel.read_edf(filepath)
            file = {"filepath": filepath,
            "signals": signals,
            "signal_headers": signal_headers,
            "header": header}
            files.append(file)
            
    return files

def save_edf_file(file, output_path: str="data/filtered/"):
    filepath = Path(file["filepath"])
    # Make sure the physical boundaries are set properly
    for signal, header in zip(file["signals"], file["signal_headers"]):
        header["'physical_max'"] = signal.max()
        header["'physical_min'"] = signal.min()
    
    return pyedflib.highlevel.write_edf(output_path + filepath.stem + filepath.suffix, file["signals"],
                                        file["signal_headers"], file["header"], digital=False)