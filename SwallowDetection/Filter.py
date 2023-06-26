import numpy as np
import scipy.signal
import math

class Filter:

    def __init__(self, sample_rate):

        if sample_rate == 1000 or sample_rate == 4000:
            self.__sample_rate = sample_rate
        else:
            print("Sample Rate not Supported only 1000Hz or 4000Hz" )



    ####################################################################################################################
    ## filter functions

    def filter_butter_TP(self, data, fg, order):
        b3, a3 = scipy.signal.butter(order, fg / (self.__sample_rate/2), output='ba')
        bi_butter = scipy.signal.lfilter(b3, a3, data, axis=- 1, zi=None)
        return bi_butter

    def filter_bi_segmentation(self, data, fg):
        b, a = scipy.signal.butter(3, fg / (self.__sample_rate/2), output='ba')
        bi_butter = scipy.signal.lfilter(b, a, data, axis=- 1, zi=None)
        return bi_butter

    def filter_emg(self, data):

        b, a = scipy.signal.butter(3, 30 / (self.__sample_rate/2), btype='high', analog=False, output='ba')
        emg_hp = scipy.signal.lfilter(b, a, data, axis=- 1, zi=None)

        emg_ds = self.__despike(emg_hp)
        emg_ds = self.__lynnFuerstNotch(emg_ds)
        emg_filtered = self.__whitening(emg_ds)

        return emg_filtered

    def filter_emg_without_HP(self, data): # use for long term measurement (already HP filtered)

        emg_ds = self.__despike(data)
        emg_ds = self.__whitening(emg_ds)
        emg_filtered = self.__lynnFuerstNotch(emg_ds)

        if self.__sample_rate == 4000:
            b, a = scipy.signal.butter(3, 300 / (self.__sample_rate/2), output='ba')
            clean = scipy.signal.lfilter(b, a, emg_filtered, axis=- 1, zi=None)
        else:
            clean = emg_filtered

        return clean

    def filter_at(self, data):
        b3, a3 = scipy.signal.butter(3, 10 / (self.__sample_rate/2), output='ba')
        bi_butter = scipy.signal.lfilter(b3, a3, data, axis=- 1, zi=None)
        return bi_butter

    def __lynnFuerstNotch(self,data):

        emg_butter = data

        for i in range(1, 6):
            w = (i * 50) / self.__sample_rate * 2 * math.pi
            r = 0.995

            b = [1, -2 * math.cos(w), 1]
            a = [1, -2 * r * math.cos(w), r * r]
            emg_butter = scipy.signal.lfilter(b, a, emg_butter, axis=- 1, zi=None)

        return emg_butter

    def __whitening(self, data):

        if self.__sample_rate == 4000: # including HP 30Hz order 2; TP 300Hz order 3

            b = [0.047831, -0.0740789, -0.1387778, 0.2266762, 0.1335115, -0.2316668, -0.0425647, 0.0790696]
            # ----------------------------------------------------------------------------------------------
            a = [-0.3054561, 1.4727165, -2.2774294, 0.1966855, 3.0665438, -3.1527876, 1]

            a = np.flipud(a)
            b = np.flipud(b)

        else:
            print("sample rate not supported by whitening")
            return 0

        emg_white = scipy.signal.lfilter(b, a, data, axis=- 1, zi=None)
        return emg_white


    def __despike(self, data):

        despiked = np.zeros(data.shape)
        correction_mode = False
        correction_value = 0
        counter = 0
        lenght_diff_buffer = int(0.5 * self.__sample_rate)
        diffs = np.zeros((lenght_diff_buffer,))
        diffs_counter = 0
        border = 0.01
        sample_border = math.ceil(0.01  * self.__sample_rate)

        for i in range(1, data.shape[0]):

            diff = data[i] - data[i - 1]
            diffs[diffs_counter] = diff
            diffs_counter = diffs_counter + 1

            if diffs_counter == lenght_diff_buffer:
                var = np.var(diffs)
                diffs_counter = 0
                border = (var * 144)

            if diff * diff > border and correction_mode == False:
                # print("Correction Mode On: " + str(i))
                correction_mode = True
                correction_value = data[i - 1]
                counter = 0

            despiked[i] = data[i]
            if correction_mode:
                despiked[i] = correction_value
                counter = counter + 1

            if (abs(data[i]) < abs(correction_value) or counter > sample_border) and correction_mode == True:
                correction_mode = False

        return despiked




    def __readEDF(self, pathAndFile):

        # if __name__ == '__main__':
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
                freqs = freqs.append(
                    pd.DataFrame({'Frequenzy': sampleFreqs[i], 'Count': 1, 'SampleCount': f.readSignal(i).shape[0]},
                                 index=np.arange(freqs.shape[0], freqs.shape[0] + 1)))

        label = f.getSignalLabels()

        data = {name: [0, 0] for name in label}

        for i in range(0, np.size(label)):
            data[label[i]] = f.readSignal(i)

        annotationData = f.readAnnotations()
        annotations = pd.DataFrame(columns=['ann_text', 'ann_time', 'ann_dur'])

        for i in range(0, annotationData[0].shape[0]):
            annotations = annotations.append(pd.DataFrame(
                {'ann_text': annotationData[2][i], 'ann_time': annotationData[0][i], 'Duration': annotationData[1][i]},
                index=np.arange(annotations.shape[0], annotations.shape[0] + 1)))

        f._close()
        del f
        return data, annotations