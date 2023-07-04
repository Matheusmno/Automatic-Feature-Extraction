from SwallowDetection.SwallowAnntations import get_swallow_annotations
import EDF_wrapper
import os
import numpy as np
from pathlib import Path
import re
import pandas as pd

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute


def add_swallow_annotations(file, output_path:str="data/annotated/"):
    
    try:
        times, annotations = get_swallow_annotations(file['filepath'])
    except:
        print(f"File {file['filepath']} failed to get swallow annotations.")
    
    os.makedirs(output_path, exist_ok=True)

    for time, annotation in zip(times, annotations):
        file['header']['annotations'].append([time, -1, annotation])
        
    file['header']['annotations'].sort(key=lambda x: x[0])
    # Save edited edf file
    EDF_wrapper.save_edf_file(file, output_path=output_path)
    
def add_swallow_annotations_to_files(files: list, output_path:str="data/annotated/"):
    
    # Extract annotations from signal
    ann = []    
    for edf_file in np.asarray([file['filepath'] for file in files]):
        try:
            ann.append(get_swallow_annotations(edf_file))
        except:
            print(f"File {edf_file} failed to get swallow annotations.")
    
    os.makedirs(output_path, exist_ok=True)
    
    for file, (times, annotations) in zip(files, ann):
        # Add extracted annotations to file's annotation list
        for time, annotation in zip(times, annotations):
            file['header']['annotations'].append([time, -1, annotation])
            
        file['header']['annotations'].sort(key=lambda x: x[0])
        # Save edited edf file
        EDF_wrapper.save_edf_file(file, output_path=output_path)
        

def compute_time(sampling_frequency, signal_array):
        # Calculate the time array based on the length of the signal array and the sampling frequency
        total_samples = len(signal_array)
        time_array = np.arange(total_samples) / sampling_frequency
        
        return time_array
    
def find_first_element(list_data, condition):
        for element in list_data:
            if condition(element):
                return element
        return None

def crop_signals_array(start_time, stop_time, file):
        cropped_signals = []
        for channel, signal in enumerate(file["signals"]):
            sr = file['signal_headers'][channel]['sample_rate']
            start_idx = round(start_time * sr)
            stop_idx = round(stop_time * sr) + 1
            time_array = compute_time(sr, signal)
            cropped_signals.append((time_array[start_idx: stop_idx], np.array(signal[start_idx: stop_idx])))
            
        return list(zip(file['signal_headers'], np.array(cropped_signals)))
    
def create_annotations_df(file, df_type='general', fileList=False):

    if df_type == 'general':
        match_pattern = "[ctp]_"
    else:
        match_pattern = "[cs]_"

    general = list(filter(lambda x : re.match(match_pattern, x[-1]), file["header"]["annotations"]))

    id_rows = {"set": [], "subject": [], "category": [], "sample_name": [],
            "start_time": [], "stop_time": [],
             }
    
    signal_rows = {"id": [],
             "data_label": [],
             "time": [],
             "signal": []
             }
    
    cat = '-'
    id = -1
    for i, row in enumerate(general):
        time, _, desc = row
        s = desc.split("_")
        t, sample, event = s
        
        if t == "c":
            if event == "start":
                _, cat, _ = s
            else:
                cat = '-'
            
        else:
            if event == "start":
                start_time = time
                stop_time, _, _ = find_first_element(general[i:], lambda x: x[-1] == f"{t}_{sample}_stop")
                if not fileList:
                    id += 1
                    signals = crop_signals_array(start_time, stop_time, file)
                                            
                    id_rows["set"].append(1)
                    id_rows["subject"].append(Path(file["filepath"]).stem)
                    id_rows["category"].append(cat)
                    id_rows["sample_name"].append(s[1])
                    id_rows["start_time"].append(start_time)
                    id_rows["stop_time"].append(stop_time)
                    
                    for h, sigs in signals:
                        signal_rows["id"].append(id)
                        signal_rows["data_label"].append(h['label'])
                        signal_rows["time"].append(sigs[0])
                        signal_rows["signal"].append(sigs[1])

    main_df = pd.DataFrame(id_rows)
    
    signals_df = pd.DataFrame(signal_rows)
    
    df = main_df.merge(signals_df, left_index=True, right_on='id')
     
    return df

def extract_features_from_annotations(annotations_df, extraction_settings):
    annotations_df_1NF = annotations_df.explode(['time', 'signal'])
    annotations_df_1NF['id'] = annotations_df_1NF.index

    # Cast columns to meet tsFRESH's requirements
    annotations_df_1NF["time"] = annotations_df_1NF["time"].astype(float)
    annotations_df_1NF["signal"] = annotations_df_1NF["signal"].astype(float)

    X = extract_features(annotations_df_1NF, column_id='id', column_sort='time',
                     column_kind=None, column_value='signal',
                     default_fc_parameters=extraction_settings,
                     impute_function=impute
                     )
    
    df = annotations_df.merge(X, left_on='id', right_index=True)
    df.drop(['id', 'time', 'signal'], axis=1, inplace=True)

    return df
