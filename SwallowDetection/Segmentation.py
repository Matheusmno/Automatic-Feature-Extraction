import numpy as np
import pyedflib
import scipy.signal
import pandas as pd
import math
from SwallowDetection.Filter import Filter
import matplotlib.pyplot as plt


class Segmentation:

    def __init__(self, data_file, bi_diff, delta_t, fg_bi, remove_double_swallows, desired_sample_frequency):

        self.__sample_frequency = 0 # is set during edf read
        self.__turningPointDifference = bi_diff
        self.__time_window_assignment = delta_t
        self.__remove_double_swallows = remove_double_swallows

        data, annotations = self.__readEDF(data_file) # sets self.__sample_frequency of data
        self.__annotations = annotations
        self.__desired_sample_frequency = desired_sample_frequency

        filter = Filter(self.__desired_sample_frequency)

        if "Breathing" in data.keys():
            self.__respiration = data["Breathing"]

        if self.__sample_frequency == self.__desired_sample_frequency: # long term measurements
            if "BI1" in data.keys():
                self.__bi_data = data["BI1"]
                self.__emg_data = data["EMG1"]
            elif "BI 1" in data.keys():
                self.__bi_data = data["BI 1"]
                self.__emg_data = data["EMG 1"]

        else:

            if "BI1" in data.keys():
                self.__bi_data = scipy.signal.decimate(data["BI1"],int(self.__sample_frequency / self.__desired_sample_frequency), n=None, axis=- 1, zero_phase=False)
                self.__emg_data = scipy.signal.decimate(data["EMG1"],int(self.__sample_frequency / self.__desired_sample_frequency), n=None, axis=- 1, zero_phase=False)
            elif "BI 1" in data.keys():
                self.__bi_data = scipy.signal.decimate(data["BI 1"],int(self.__sample_frequency / self.__desired_sample_frequency), n=None, axis=- 1, zero_phase=False)
                self.__emg_data = scipy.signal.decimate(data["EMG 1"],int(self.__sample_frequency / self.__desired_sample_frequency), n=None, axis=- 1, zero_phase=False)

        # filter and downsampling of BI
        self.__bi_data_filtered = filter.filter_bi_segmentation(self.__bi_data, fg_bi)
        self.__sample_frequency_bi_segmentation = 100
        sample_factor = int(self.__sample_frequency / self.__sample_frequency_bi_segmentation)
        self.__bi_data_filtered_100Hz = self.__bi_data_filtered[0:-1:sample_factor]


        up, down = self.__turningPointSegments()
        self.__segments_up = np.asarray(up)
        self.__segments_down = np.asarray(down)

        self.__defineTrueAndFalseSegmentationPoints()
        self.__calcPrecedingMaxima()



    def getTrueSegmentationPoints(self):
        return self.__true_segmentation_points

    def getFalseSegmentationPoints(self):
        return self.__false_segmentation_points

    def getAnnotations(self):
        return self.__annotations

    def getTimeDiffrencesOfSwallows(self):
        return self.__true_segmentation_points_time_diffs

    def getSegmentationPointsDown(self):
        return self.__segments_down

    def getSegmentationPointsUp(self):
        return self.__segments_up

    def getCleanedBiData(self):
        return self.__bi_data_filtered

    def getCleanedBiData100Hz(self):
        return self.__bi_data_filtered_100Hz

    def getEmgData(self):
        return self.__emg_data

    def getBiData(self):
        return self.__bi_data

    def getRespirationData(self):
        return self.__respiration

    def getSampleRate(self):
        return self.__sample_frequency

    def getTruePrecedingMaxima(self):
        return self.__true_preceding_maxima

    def getFalsePrecedingMaxima(self):
        return self.__false_preceding_maxima

    def getDeltaTimePrecedingMaxima(self):
        return self.__time_differences_preceding_maxima

    def getNonePrecedingMaxima(self):
        return self.__non_preceding_maxima

    def getNumberOfNoPrecedingMaxima(self):
        return self.__number_no_preceding_maxima

    ###################################################################################################################
    ## segmentation

    def __calcPrecedingMaxima(self): # for analysis of the timing
        true_ups = []
        index_list = []
        time_diffs = []
        no_preceding_maxima = 0

        for s in self.__true_segmentation_points:
            diffs_to_max = self.__segments_up - s
            ind = np.where(diffs_to_max < 0)[0]

            if ind.shape[0] > 0:
                true_ups.append(self.__segments_up[ind[-1]])
                index_list.append(ind[-1])
                time_diffs.append(diffs_to_max[ind[-1]])
            else:
                no_preceding_maxima = no_preceding_maxima + 1


        ##################################

        false_ups = []
        for s in self.__false_segmentation_points:
            diffs_to_max = self.__segments_up - s
            ind = np.where(diffs_to_max < 0)[0]

            if ind.shape[0] > 0:
                false_ups.append(self.__segments_up[ind[-1]])
                time_diffs.append(diffs_to_max[ind[-1]])
            else:
                no_preceding_maxima = no_preceding_maxima + 1

        self.__true_preceding_maxima = np.asarray(true_ups)
        self.__false_preceding_maxima = np.asarray(false_ups)
        self.__non_preceding_maxima = np.delete(self.__segments_up, index_list)
        self.__time_differences_preceding_maxima = np.asarray(time_diffs)
        self.__number_no_preceding_maxima = no_preceding_maxima

    def __defineTrueAndFalseSegmentationPoints(self):

        true_seg = []
        true_seg_diffs = []
        swallow_times = []
        remove_swallow_indices = []
        count_removes = 0

        for i in range(0, self.__annotations.shape[0]):

            if self.__annotations['ann_text'][i] == "Swallow":
                swallow_times.append(self.__annotations['ann_time'][i] * self.__sample_frequency_bi_segmentation)

        swallow_times = np.asarray(swallow_times, dtype=int)
        segments = np.copy(self.__segments_down)
        #segments_help = np.copy(self.__segments_down)

        for i in range(0, swallow_times.shape[0]):

            diffs = np.zeros(segments.shape)
            for j in range(0, segments.shape[0]):
                diffs[j] = (segments[j] - swallow_times[i])


            if len(diffs) > 0 and diffs[np.where(diffs >= 0)].shape[0] > 0:
                min_diffs = diffs[np.where(diffs >= 0)] # only minima after swallow anotation
                min_diffs = min_diffs[np.where(min_diffs <= int(self.__time_window_assignment * self.__sample_frequency_bi_segmentation))]

                if min_diffs.shape[0] > 0:

                    if self.__remove_double_swallows:
                        min_diffs = np.asarray(min_diffs[0])
                        min_index = np.where(diffs == min_diffs)[0]
                    else:
                        if len(min_diffs) > 1:
                            print("Not Removing doubles not checked!!")

                        min_diffs = np.asarray(min_diffs)
                        min_index = np.where(diffs == min_diffs)[0]

                    #if self.__remove_double_swallows:
                    #    remove_swallow_indices = np.append(remove_swallow_indices, np.where(diffs == min_diffs)[0] + count_removes)
                    #else:
                    #    remove_swallow_indices = np.append(remove_swallow_indices, np.where(diffs == min_diffs[0])[0] + count_removes)


                    #if min_diffs >= 0 and min_diff <= int(self.__time_window_assignment * self.__sample_frequency_bi_segmentation):

                    true_seg.extend(segments[min_index])
                    segments = np.delete(segments, min_index)
                    true_seg_diffs.extend(diffs[min_index])
                    #remove_swallow_indices = np.append(remove_swallow_indices, min_index + count_removes)
                    #count_removes = count_removes +1


        true_seg = np.asarray(true_seg, dtype=int)
        #remove_swallow_indices = np.asarray(remove_swallow_indices)
        #if true_seg.shape[0] != swallow_times.shape[0]:
        #    print("Swallows Detected: " +str(true_seg.shape[0]))
        #    print("Swallows: " + str(swallow_times.shape[0]))

        self.__true_segmentation_points = true_seg
        self.__false_segmentation_points = segments
        self.__true_segmentation_points_time_diffs = np.asarray(true_seg_diffs)


    def __turningPointSegments(self):

        segments_up = []
        segments_down = []

        last_turningpoint_type = 0 # 0 not initialized, 1 maxima, 2 minima
        last_turningpoint = 0
        data = self.__bi_data_filtered_100Hz

        for i in range(2, data.shape[0]):

            if data[i - 2] > data[i - 1] and data[i] > data[i - 1]:

                if last_turningpoint - data[i - 1] > self.__turningPointDifference and last_turningpoint_type == 1:
                    #last_turningpoint = data[i - 1]
                    segments_down.append(i)
                    last_turningpoint_type = 2
                elif last_turningpoint_type == 0:
                    last_turningpoint_type = 2

            elif data[i - 2] < data[i - 1] and data[i] < data[i - 1]:

                last_turningpoint = data[i - 1]
                segments_up.append(i)
                last_turningpoint_type = 1

        return segments_up, segments_down


    ####################################################################################################################
    ## filter functions

    def __readEDF(self, pathAndFile):

        # if __name__ == '__main__':
        #print(pathAndFile)
        f = pyedflib.EdfReader(pathAndFile)
        n = f.signals_in_file
        signal_labels = f.getSignalLabels()

        sampleFreqs = f.getSampleFrequencies()

        self.__sample_frequency = sampleFreqs[0]

        # create list containing all different samplingRates
        freqs = pd.DataFrame(columns=['Frequenzy', 'Count', 'SampleCount'])
        freqs.loc[0, 'Frequenzy'] = sampleFreqs[0]
        freqs.loc[0, 'Count'] = 0
        freqs.loc[0, 'SampleCount'] = f.readSignal(0).shape[0]

        for i in range(0, np.size(sampleFreqs)):
            checker = True
            for row in freqs.iterrows():
                d = 1
                if row[1].loc['Frequenzy'] == sampleFreqs[i]:
                    checker = False
                    row[1].loc['Count'] = row[1].loc['Count'] + 1

            if checker:
                freqs = pd.concat([freqs,
                    pd.DataFrame({'Frequenzy': sampleFreqs[i], 'Count': 1, 'SampleCount': f.readSignal(i).shape[0]},
                                 index=np.arange(freqs.shape[0], freqs.shape[0] + 1))])

        label = f.getSignalLabels()

        data = {name: [0, 0] for name in label}

        for i in range(0, np.size(label)):
            data[label[i]] = f.readSignal(i)

        annotationData = f.readAnnotations()
        annotations = pd.DataFrame(columns=['ann_text', 'ann_time', 'ann_dur'])

        # for i in range(0, annotationData[0].shape[0]):
        #     annotations = annotations.append(pd.DataFrame(
        #         {'ann_text': annotationData[2][i], 'ann_time': annotationData[0][i], 'Duration': annotationData[1][i]},
        #         index=np.arange(annotations.shape[0], annotations.shape[0] + 1)))
        
        for i in range(0, annotationData[0].shape[0]):
            annotations = pd.concat([annotations, 
                                    pd.DataFrame({'ann_text': annotationData[2][i], 'ann_time': annotationData[0][i], 'ann_dur': annotationData[1][i]},
                                                 index=np.arange(annotations.shape[0], annotations.shape[0] + 1))])

        f._close()
        del f
        return data, annotations