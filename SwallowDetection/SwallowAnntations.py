import sys
sys.path.insert(1, 'SwallowDetection')
from FeatureCalculator import FeatureCalculator
from sklearn import preprocessing
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_swallow_annotations(edf_file):

    ini_file = "SwallowDetection/feature_calculation_params_set8.ini"

    sample_rate = 4000
    only_sax = False
    set_index = 0
    file_meta = [0, 0, 0]
    calculator = FeatureCalculator(edf_file, file_meta, set_index, sample_rate, only_sax)

    calculator.setIniFile(ini_file)
    check = calculator.initialize()

    if check == False:
        print("initialization error")
        return 0


    header = calculator.getFeatureHeader()
    feature = calculator.get_feature()
    maxima = calculator.get_start_of_swallows()


    header.extend(["File", "Subject", "Set"])
    data = pd.DataFrame(data=feature, columns=header)

    drop_list = ['Sample', 'Label', 'File', 'Set', 'Subject']
    data_pred = data.drop(drop_list, axis=1)

    # predict label
    loaded_rf = joblib.load("./SwallowDetection/rf_classifer.joblib")
    predictions = loaded_rf.predict(data_pred.values)

    # create annotations
    ind_swallow_min = data[predictions.astype(bool)]['Sample'].values.astype(int)

    # find swallow start
    ind_swallow_start = []
    for ind in ind_swallow_min:
        ind_swallow_start.append(maxima[np.where(maxima < ind)[0][-1]])

    ind_swallow_start = np.asarray(ind_swallow_start).astype(int)

    # find swallow end
    bi_data = calculator.get_cleaned_bi_100Hz()
    dif_percantage = 0.8
    max_search = 200  # two seconds

    ind_swallow_stop = []
    for i in range(0, ind_swallow_start.shape[0]):

        bi_diff = (bi_data[ind_swallow_start[i]] - bi_data[ind_swallow_min[i]]) * dif_percantage
        bi_min = bi_data[ind_swallow_min[i]]

        # check increasing BI
        for j in range(ind_swallow_min[i], bi_data.shape[0]):

            if bi_data[j] - bi_min >= bi_diff or max_search <= (j - ind_swallow_min[i]):
                ind_swallow_stop.append(j)
                break

    # last swallow stop not found at measurement end
    if len(ind_swallow_stop) != ind_swallow_start.shape[0]:
        ind_swallow_stop.append(bi_data.shape[0])

    ind_swallow_stop = np.asarray(ind_swallow_stop).astype(int)

    # shuffle annotations
    indices = []
    label = []
    for i in range(0, ind_swallow_start.shape[0]):
        indices.append(ind_swallow_start[i])
        indices.append(ind_swallow_stop[i])

        label.append("s_swallow_start")
        label.append("s_swallow_stop")

    plt.figure()
    plt.plot(bi_data)
    plt.plot(ind_swallow_start, bi_data[ind_swallow_start], '>', color = 'r')
    plt.plot(ind_swallow_min, bi_data[ind_swallow_min], 'v', color = 'r')
    plt.plot(ind_swallow_stop, bi_data[ind_swallow_stop], '<', color = 'r')
    #plt.show()

    times = np.asarray(indices) / 100
    label = np.asarray(label)
    annotations = (times, label)
    return annotations

if __name__ == '__main__':

    edf_files = []
    edf_files.append("/Users/matheusnoschang/Programming/Repos/Automatic-Feature-Extraction/SwallowDetection/1-1-Bewegung.bdf")
    edf_files.append("/Users/matheusnoschang/Programming/Repos/Automatic-Feature-Extraction/SwallowDetection/1-7-Schlucktest_Leitfaehigkeit.bdf")
    edf_files.append("/Users/matheusnoschang/Programming/Repos/Automatic-Feature-Extraction/SwallowDetection/1-8-Bewegung.bdf")

    for edf_file in np.asarray(edf_files):
        annotations = get_swallow_annotations(edf_file)
    plt.show()