from annotations_validation import check_annotations
import EDF_wrapper
from pathlib import Path
from tsfresh.feature_extraction import EfficientFCParameters
from annotations_extraction import add_swallow_annotations_to_files, create_annotations_df, extract_features_from_annotations

input_folder = 'data/'

edf_files = EDF_wrapper.read_files_from_dir(input_folder + 'edf/', load_files=True)

add_swallow_annotations_to_files(edf_files, output_path=input_folder + 'annotated/')

annotated_files = EDF_wrapper.read_files_from_dir(input_folder + 'annotated/', load_files=True)

for ann_file in annotated_files:
    # Make sure the annotations are always lower-case
    ann_file["header"]["annotations"] = list(map(lambda x: [x[0], x[1], x[2].lower()], ann_file["header"]["annotations"]))

    if check_annotations(ann_file):

        general_df = create_annotations_df(ann_file, 'general')
        swallows_df = create_annotations_df(ann_file, 'swallows')

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

        df = extract_features_from_annotations(general_df, extraction_settings)

        df.to_excel(input_folder + f"xlsx/{Path(ann_file['filepath']).stem}.xlsx")
    
    else:
        print("There are no annotations that follow the expected pattern.")