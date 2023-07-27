from SwallowDetection.SwallowAnntations import get_swallow_annotations
import EDF_wrapper
import annotations_validation as av
import os
import numpy as np
from pathlib import Path
import re
import pandas as pd

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import EfficientFCParameters

def add_swallow_annotations(f, output_path:str="data/annotated/"):
    
    try:
        times, annotations = get_swallow_annotations(f['filepath'])
    except:
        print(f"File {f['filepath']} failed to get swallow annotations.")
    
    os.makedirs(output_path, exist_ok=True)

    for time, annotation in zip(times, annotations):
        f['header']['annotations'].append([time, -1, annotation])
        
    f['header']['annotations'].sort(key=lambda x: x[0])
    # Save edited edf file
    EDF_wrapper.save_edf_file(f, output_path=output_path)
    
def add_swallow_annotations_to_files(files: list, output_path:str="data/annotated/"):
    
    # Extract annotations from signal
    ann = []    
    for edf_file in np.asarray([f['filepath'] for f in files]):
        try:
            ann.append(get_swallow_annotations(edf_file))
        except:
            print(f"File {edf_file} failed to get swallow annotations.")
    
    os.makedirs(output_path, exist_ok=True)
    
    for f, (times, annotations) in zip(files, ann):
        # Add extracted annotations to file's annotation list
        for time, annotation in zip(times, annotations):
            f['header']['annotations'].append([time, -1, annotation])
            
        f['header']['annotations'].sort(key=lambda x: x[0])
        # Save edited edf file
        EDF_wrapper.save_edf_file(f, output_path=output_path)
        

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

def crop_signals_array(start_time, stop_time, f):
        cropped_signals = []
        for channel, signal in enumerate(f["signals"]):
            sr = f['signal_headers'][channel]['sample_rate']
            start_idx = round(start_time * sr)
            stop_idx = round(stop_time * sr) + 1
            time_array = compute_time(sr, signal)
            cropped_signals.append((time_array[start_idx: stop_idx], np.array(signal[start_idx: stop_idx])))
            
        return list(zip(f['signal_headers'], np.array(cropped_signals)))
    
def create_annotations_df(f, df_type='general', fileList=False, signal_labels_to_extract=None):

    if df_type == 'general':
        match_pattern = "[ctp]_([^\s_]+)_(start|stop)"
    else:
        match_pattern = "[cs]_([^\s_]+)_(start|stop)"

    general = list(filter(lambda x : re.match(match_pattern, x[-1]), f["header"]["annotations"]))

    id_rows = {"filename": [], "set": [], "subject": [], "category": [], "sample_name": [],
            "start_time": [], "stop_time": [],
             }
    
    signal_rows = {"ann_id": [],
             "data_label": [],
             "time": [],
             "signal": []
             }
    
    cat = ""
    id = -1
    for i, row in enumerate(general):
        time, _, desc = row
        s = desc.split("_")
        t, sample, event = s
        
        if t == "c":
            if event == "start":
                _, cat, _ = s
            else:
                cat = ""
            
        else:
            if event == "start":
                start_time = time
                stop_time, _, _ = find_first_element(general[i:], lambda x: x[-1] == f"{t}_{sample}_stop")
                if not fileList:
                    id += 1
                    signals = crop_signals_array(start_time, stop_time, f)
                                            
                    id_rows["filename"].append(Path(f["filepath"]).stem)
                    id_rows["set"].append(1)
                    id_rows["subject"].append(0)
                    id_rows["category"].append(cat)
                    id_rows["sample_name"].append(s[1])
                    id_rows["start_time"].append(start_time)
                    id_rows["stop_time"].append(stop_time)
                    
                    for h, sigs in signals:
                        signal_rows["ann_id"].append(id)
                        signal_rows["data_label"].append(h['label'])
                        signal_rows["time"].append(sigs[0])
                        signal_rows["signal"].append(sigs[1])
                else:
                    id += 1
                    signals = crop_signals_array(start_time, stop_time, f)
                                            
                    id_rows["filename"].append(Path(f["filepath"]).stem)
                    id_rows["set"].append(f["set"])
                    id_rows["subject"].append(f["subject"])
                    id_rows["category"].append(cat)
                    id_rows["sample_name"].append(s[1])
                    id_rows["start_time"].append(start_time)
                    id_rows["stop_time"].append(stop_time)
                    
                    for h, sigs in signals:
                        signal_rows["ann_id"].append(id)
                        signal_rows["data_label"].append(h['label'])
                        signal_rows["time"].append(sigs[0])
                        signal_rows["signal"].append(sigs[1])

    main_df = pd.DataFrame(id_rows)
    
    signals_df = pd.DataFrame(signal_rows)
    
    df = main_df.merge(signals_df, left_index=True, right_on='ann_id')
    
    if signal_labels_to_extract is None:
        return df
    return df[df.data_label.isin(signal_labels_to_extract)]

def swallow_extend(row):
    time_start_idx = np.argwhere(row['time'] == row['BI_start_time'])[0][0]
    time_end_idx = np.argwhere(row['time'] == row['BI_min'])[0][0]
    if 'BI' in row['data_label']:
        return np.trapz(row['signal'][time_start_idx:time_end_idx+1], row['time'][time_start_idx:time_end_idx+1])
    else:
        return np.nan
    
def area_under_EMG_envelope(row):
    time_start_idx = np.argwhere(row['time'] == row['BI_start_time'])[0][0]
    time_end_idx = np.argwhere(row['time'] == row['BI_min'])[0][0]
    if 'EMG' in row['data_label']:
        return np.trapz(row['signal'][time_start_idx:time_end_idx+1], row['time'][time_start_idx:time_end_idx+1])
    else:
        return np.nan

def add_basic_swallow_features(swallows_df):  
    # Swallow basic features
    if not swallows_df.empty:
        swallows_df['BI_start_time'] = swallows_df.apply(lambda row : row['time'][0] if 'BI' in row['data_label'] else np.nan, axis=1)
        swallows_df['BI_start_time'] = swallows_df.groupby('ann_id')['BI_start_time'].transform('first') # Fill values for all signals belonging to the same annotation

        #swallows_df['EMG_start_time'] = swallows_df['BI_start_time'] - 0.366
        swallows_df['BI_stop_time'] = swallows_df.apply(lambda row : row['time'][-1] if 'BI' in row['data_label'] else np.nan, axis=1)
        swallows_df['BI_stop_time'] = swallows_df.groupby('ann_id')['BI_stop_time'].transform('first') # Fill values for all signals belonging to the same annotation

        swallows_df['BI_min'] = swallows_df.apply(lambda row : row['time'][row['signal'].argmin()] if 'BI' in row['data_label'] else np.nan, axis=1)
        swallows_df['BI_min'] = swallows_df.groupby('ann_id')['BI_min'].transform('first') # Fill values for all signals belonging to the same annotation

        swallows_df['elevation_duration'] = swallows_df['BI_min'] - swallows_df['BI_start_time']

        swallows_df['elevation'] = swallows_df.apply(lambda row : row['signal'][0] - row['signal'][row['signal'].argmin()] if 'BI' in row['data_label'] else np.nan, axis=1)
        swallows_df['elevation'] = swallows_df.groupby('ann_id')['elevation'].transform('first') # Fill values for all signals belonging to the same annotation

        swallows_df['elevation_speed'] = swallows_df.apply(lambda row : (row['signal'][0] - row['signal'][row['signal'].argmin()])/(row['time'][0] - row['time'][row['signal'].argmin()]) if 'BI' in row['data_label'] else np.nan, axis=1)
        swallows_df['elevation_speed'] = swallows_df.groupby('ann_id')['elevation_speed'].transform('first') # Fill values for all signals belonging to the same annotation

        swallows_df['swallow_duration'] = swallows_df['BI_stop_time'] - swallows_df['BI_start_time']

        swallows_df['swallow_extend'] = swallows_df.apply(swallow_extend, axis=1)
        swallows_df['swallow_extend'] = swallows_df.groupby('ann_id')['swallow_extend'].transform('first') # Fill values for all signals belonging to the same annotation

        # Check why negative values
        swallows_df['area_under_EMG_envelope'] = swallows_df.apply(area_under_EMG_envelope, axis=1)
        swallows_df['area_under_EMG_envelope'] = swallows_df.groupby('ann_id')['area_under_EMG_envelope'].transform('first') # Fill values for all signals belonging to the same annotation


        #swallows_df['signal_area'] = swallows_df.apply(lambda row : np.trapz(row['signal'], row['time']), axis=1)
        
        swallows_df = swallows_df.groupby('ann_id').first()
        
        swallows_df.drop(['start_time','stop_time','time','signal'], axis=1, inplace=True)
        
        return swallows_df
    
    else:    
        print("The annotations dataframe is empty.")
        return pd.DataFrame()

def add_basic_general_features(general_df):
    # General basic features
    if not general_df.empty:
        general_df['span'] = general_df.apply(lambda row : row['signal'].max() - row['signal'].min(), axis=1)
        general_df['area'] = general_df.apply(lambda row : np.trapz(row['signal'], row['time']), axis=1)
        general_df['duration'] = general_df['stop_time'] - general_df['start_time']
        general_df['start_value'] = general_df['signal'][0]
        general_df['stop_value'] = general_df['signal'][-1]
        general_df['span_start_stop_value'] = general_df['start_value'] - general_df['stop_value']
        general_df['IQR'] = general_df.apply(lambda row : np.subtract(*np.percentile(row['signal'], [75, 25])))
        return general_df
    else:
        print("The annotations dataframe is empty.")
        return general_df
                

def add_tsfresh_features(annotations_df):
    
    extraction_settings = EfficientFCParameters()
    
    del extraction_settings['index_mass_quantile']
    del extraction_settings['time_reversal_asymmetry_statistic']
    del extraction_settings['cid_ce']
    del extraction_settings['symmetry_looking']
    del extraction_settings['agg_autocorrelation']
    del extraction_settings['cwt_coefficients']
    del extraction_settings['spkt_welch_density']
    del extraction_settings['ar_coefficient']
    del extraction_settings['change_quantiles']
    del extraction_settings['fft_coefficient']
    del extraction_settings['fft_aggregated']
    del extraction_settings['agg_linear_trend']
    del extraction_settings['augmented_dickey_fuller']
    del extraction_settings['ratio_beyond_r_sigma']
    del extraction_settings['fourier_entropy']
    del extraction_settings['permutation_entropy']
    del extraction_settings['lempel_ziv_complexity']
    del extraction_settings['query_similarity_count']
    del extraction_settings['number_crossing_m']
    del extraction_settings['large_standard_deviation']
    extraction_settings['quantile'] = extraction_settings['quantile'][-2:]
    
    if not annotations_df.empty:
        annotations_df_1NF = annotations_df.explode(['time', 'signal'])

        # Cast columns to meet tsFRESH's requirements
        annotations_df_1NF["time"] = annotations_df_1NF["time"].astype(float)
        annotations_df_1NF["signal"] = annotations_df_1NF["signal"].astype(float)

        X = extract_features(annotations_df_1NF, column_id='ann_id', column_sort='time',
                        column_kind=None, column_value='signal',
                        default_fc_parameters=extraction_settings,
                        impute_function=impute
                        )

        df = annotations_df.merge(X, left_index=True, right_index=True) # Join annotations with extracted features
        df.drop(['time', 'signal'], axis=1, inplace=True)
        return df
    else:
        print("The annotations dataframe is empty.")

    


def save_features_excel_files(file_list, csv=False, output_path="data/xlsx/", signal_labels_to_extract=None):

    dir_general_features = pd.DataFrame()
    dir_basic_swallow_features = pd.DataFrame()
    dir_tsfresh_swallow_features = pd.DataFrame()

    for edf_file in file_list:
        if av.check_annotations(edf_file):
            # Create general features excel file
            general_df = create_annotations_df(edf_file, 'general', fileList=csv, signal_labels_to_extract=signal_labels_to_extract)
            general_df_fe = add_basic_general_features(general_df)
            general_df_fe = add_tsfresh_features(general_df_fe)
            
            # Create baisc swallow features and tsfresh features excel files
            swallows_df = create_annotations_df(edf_file, 'swallows', fileList=csv, signal_labels_to_extract=signal_labels_to_extract)
            basic_swallows_df_fe = add_basic_swallow_features(swallows_df.copy())
            #basic_swallows_df_fe = basic_swallows_df_fe
            ts_fresh_swallows_df_fe = add_tsfresh_features(swallows_df)
            if general_df_fe is not None:
                dir_general_features = pd.concat([dir_general_features, general_df_fe], axis=0)
                #general_df_fe.to_excel(f"data/xlsx/{Path(edf_file['filepath']).stem}_general_features.xlsx", index=False)
            if basic_swallows_df_fe is not None:
                dir_basic_swallow_features = pd.concat([dir_basic_swallow_features, basic_swallows_df_fe], axis=0)
                #swallows_df_fe.to_excel(f"data/xlsx/{Path(edf_file['filepath']).stem}_basic_swallow_features.xlsx", index=False)
            if ts_fresh_swallows_df_fe is not None:
                dir_tsfresh_swallow_features = pd.concat([dir_tsfresh_swallow_features, ts_fresh_swallows_df_fe], axis=0)
                #swallows_df_fe.to_excel(f"data/xlsx/{Path(edf_file['filepath']).stem}_ts_fresh_swallow_features.xlsx", index=False)

    if not dir_general_features.empty:
        dir_general_features.drop(['ann_id'], axis=1, inplace=True)
        dir_general_features.to_excel(output_path + "/general_features.xlsx", index=False)
    if not dir_basic_swallow_features.empty:
        dir_basic_swallow_features.drop(['data_label'], axis=1, inplace=True)
        dir_basic_swallow_features.to_excel(output_path + "/basic_swallow_features.xlsx", index=False)
    if not dir_tsfresh_swallow_features.empty:
        dir_tsfresh_swallow_features.drop(['ann_id'], axis=1, inplace=True)
        dir_tsfresh_swallow_features.to_excel(output_path + "/tsfresh_swallow_features.xlsx", index=False)
        
def extract_features_from_filelist(filelist_csv_path: str, data_root_path: str, output_path=None, signal_labels_to_extract=['BI1', 'EMG1']):
    if output_path is None:
        output_path = data_root_path + "data/features"
    files = EDF_wrapper.read_files_from_csv_filelist(data_root_path, filelist_csv_path)
    os.makedirs(output_path, exist_ok=True)
    save_features_excel_files(files, csv=True, output_path=output_path,
                                    signal_labels_to_extract=signal_labels_to_extract)
    
def extract_features_from_folder(folder_path: str, output_path=None, signal_labels_to_extract=['BI1', 'EMG1']):
    if output_path is None:
        output_path = "data/features"
    files = EDF_wrapper.read_files_from_dir(folder_path)
    os.makedirs(output_path, exist_ok=True)
    save_features_excel_files(files, output_path=output_path,
                                    signal_labels_to_extract=signal_labels_to_extract
                                    )