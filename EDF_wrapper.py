import pyedflib
from pathlib import Path
import os
import pandas as pd
import numpy as np


def read_files_from_dir(directory: Path, load_files:bool=True):
    extensions = ["edf", "bdf"]
    files = []

    for file in os.scandir(directory):
        
        if file.is_file() and file.name.endswith(tuple(extensions)):
            if load_files:
                filepath = file.path
                signals, signal_headers, header = pyedflib.highlevel.read_edf(filepath)
                file = {"filepath": filepath,
                "signals": signals,
                "signal_headers": signal_headers,
                "header": header}
                files.append(file)
            else:
                filepath = file.path
                file = {"filepath": filepath,
                    "signals": None,
                    "signal_headers": None,
                    "header": None}
                files.append(file)
                
    return files

def read_files_from_csv_filelist(filelist: Path, load_files:bool=True):
    extensions = ["edf", "bdf"]
    files = []
    filelist = pd.read_csv(filelist)

    for _, file in filelist.iterrows():
        if load_files:
            filepath = file.Files
            signals, signal_headers, header = pyedflib.highlevel.read_edf(filepath)
            file = {"filepath": filepath,
            "signals": signals,
            "signal_headers": signal_headers,
            "header": header,
            "subject": file.Subject,
            "set": file.Set}
            files.append(file)
        else:
            filepath = file.Files
            file = {"filepath": filepath,
                "signals": None,
                "signal_headers": None,
                "header": None,
                "subject": None,
                "set": None}
            files.append(file)
                
    return files

def save_edf_file(file, output_path: str="data/filtered/"):
    #os.makedirs(output_path, exist_ok=True)
    filepath = Path(file["filepath"])
    # Make sure the physical boundaries are set properly
    for signal, header in zip(file["signals"], file["signal_headers"]):
        if signal.min() != signal.max():
            header['physical_max'] = signal.max()
            header['physical_min'] = signal.min()
        else:
            header['physical_max'] = signal.min() + 1 
            header['physical_min'] = signal.min()
    
    return pyedflib.highlevel.write_edf(output_path + filepath.stem + filepath.suffix, file["signals"],
                                        file["signal_headers"], file["header"], digital=False)