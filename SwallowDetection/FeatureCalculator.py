import pandas as pd
import numpy as np
from Segmentation import Segmentation
from SAXConverter import SAXConverter
from Filter import Filter
import math
import configparser
import os
from sklearn.preprocessing import StandardScaler


class FeatureCalculator:

    def __init__(self, data_file, file_meta, set, sample_frequency, only_sax):

        self.__only_sax = only_sax
        self.__file_meta = file_meta

        #self.__bow_sax_converter_bi = BOWConverter(3,3,False)
        #self.__bow_sax_converter_emg = BOWConverter(3, 3, True)

        self.__data_file = data_file

        self.__bi_sax_converter = SAXConverter(10, 16, False)
        self.__emg_sax_converter = SAXConverter(20, 16, True)

        self.__bin_emg_samples = 100 # used for Max and MaxIndex
        self.__sample_frequency = sample_frequency

        self._set = set


    def initialize(self):

        return_value = True

        self.__true_points = self.__segmentation.getTrueSegmentationPoints()
        self.__false_points = self.__segmentation.getFalseSegmentationPoints()

        self.__preceding_true_maxima = self.__segmentation.getTruePrecedingMaxima()
        self.__preceding_false_maxima = self.__segmentation.getFalsePrecedingMaxima()

        #if self.__true_points.shape[0] == 0:
        #    print("FeatureCalculator: No True Points Found")
        #    return_value = False

        #else:

        if self.__false_points[0] < self.__preceding_false_maxima[0]:
            self.__false_points = np.delete(self.__false_points, [0])

        #if self.__true_points[0] < self.__preceding_true_maxima[0]:
        #    self.__true_points = np.delete(self.__true_points, [0])

        #for i in range(0, self.__true_points.shape[0]):
        #    if self.__true_points[i] < self.__preceding_true_maxima[i]:
        #        print("FeatureCalculator: Index_TrueDown > Index_TrueUp")
        #        return_value = False

        for i in range(0, self.__false_points.shape[0]):
            if self.__false_points[i] < self.__preceding_false_maxima[i]:
                print("FeatureCalculator: Index_TrueDown > Index_TrueUp")
                return_value = False

        # prepare Data
        self.__filter = Filter(self.__sample_frequency)

        # filter emg
        emg_data = self.__segmentation.getEmgData()
        emg_data = self.__filter.filter_emg(emg_data)
        self.__emg_data = emg_data

        # filtered BI
        self.__bi_data = self.__segmentation.getCleanedBiData100Hz()

        # filter tEMG
        emg_data = self.__segmentation.getEmgData()
        self.__emgt_data = self.__filter.filter_butter_TP(emg_data, 10, 3)

        return return_value

    def setIniFile(self, file):

        if os.path.exists(file) == False:
            print("FeatureCalculator: ini-File not existent")
            return

        read_config = configparser.ConfigParser()
        read_config.read(file)


        # EMG SAX
        value = int(read_config.get("sax", "emg_before"))
        self.__emg_sax_sample_before = value
        value = int(read_config.get("sax", "emg_after"))
        self.__emg_sax_sample_after = value

        # BI SAX
        value = int(read_config.get("sax", "bi_before"))
        self.__bi_sax_sample_before = value
        value = int(read_config.get("sax", "bi_after"))
        self.__bi_sax_sample_after = value

        # reverse

        ### BI Variation ##
        # DifBI
        value = int(read_config.get("borders_reverse", "DifBI_l"))
        self.__dif_bi_rev_l = value
        value = int(read_config.get("borders_reverse", "DifBI_k"))
        self.__dif_bi_rev_k = value
        # SigBI
        value = int(read_config.get("borders_reverse", "SigBI_l"))
        self.__std_bi_rev_l = value
        value = int(read_config.get("borders_reverse", "SigBI_k"))
        self.__std_bi_rev_k = value

        ### EMGT Variation ##
        # DifEMGT
        value = int(read_config.get("borders_reverse", "DifEMGT_l"))
        self.__dif_emgt_rev_l = value
        value = int(read_config.get("borders_reverse", "DifEMGT_k"))
        self.__dif_emgt_rev_k = value
        # SigBI
        value = int(read_config.get("borders_reverse", "SigEMGT_l"))
        self.__std_emgt_rev_l = value
        value = int(read_config.get("borders_reverse", "SigEMGT_k"))
        self.__std_emgt_rev_k = value

        ### IndBI ##
        # IndMinBI
        value = int(read_config.get("borders_reverse", "IndMinBI_l"))
        self.__ind_min_bi_rev_l = value
        value = int(read_config.get("borders_reverse", "IndMinBI_k"))
        self.__ind_min_bi_rev_k = value
        # IndMaxBI
        value = int(read_config.get("borders_reverse", "IndMaxBI_l"))
        self.__ind_max_bi_rev_l = value
        value = int(read_config.get("borders_reverse", "IndMaxBI_k"))
        self.__ind_max_bi_rev_k = value

        ### IndEMGT ##
        # IndMinEMGT
        value = int(read_config.get("borders_reverse", "IndMinEMGT_l"))
        self.__ind_min_emgt_rev_l = value
        value = int(read_config.get("borders_reverse", "IndMinEMGT_k"))
        self.__ind_min_emgt_rev_k = value
        # IndMaxEMGT
        value = int(read_config.get("borders_reverse", "IndMaxEMGT_l"))
        self.__ind_max_emgt_rev_l = value
        value = int(read_config.get("borders_reverse", "IndMaxEMGT_k"))
        self.__ind_max_emgt_rev_k = value

        ### IndEMG ##
        # MaxEMG
        value = int(read_config.get("borders_reverse", "MaxEMG_l"))
        self.__max_emg_rev_l = value
        value = int(read_config.get("borders_reverse", "MaxEMG_k"))
        self.__max_emg_rev_k = value
        # IndMaxEMG1
        value = int(read_config.get("borders_reverse", "IndMaxEMG1_l"))
        self.__ind_max_emg_1_rev_l = value
        value = int(read_config.get("borders_reverse", "IndMaxEMG1_k"))
        self.__ind_max_emg_1_rev_k = value
        # IndMaxEMG2
        value = int(read_config.get("borders_reverse", "IndMaxEMG2_l"))
        self.__ind_max_emg_2_rev_l = value
        value = int(read_config.get("borders_reverse", "IndMaxEMG2_k"))
        self.__ind_max_emg_2_rev_k = value
        # IndMinEMG
        value = int(read_config.get("borders_reverse", "IndMinEMG_l"))
        self.__ind_min_emg_rev_l = value
        value = int(read_config.get("borders_reverse", "IndMinEMG_k"))
        self.__ind_min_emg_rev_k = value

        ###########################################################
        # foreward

        ### EMG Activity ###
        # RMS
        value = int(read_config.get("borders_foreward", "RMS_l"))
        self.__rms_l = value
        value = int(read_config.get("borders_foreward", "RMS_k"))
        self.__rms_k = value
        #ZCR
        value = int(read_config.get("borders_foreward", "ZCR_l"))
        self.__zcr_l = value
        value = int(read_config.get("borders_foreward", "ZCR_k"))
        self.__zcr_k = value
        #AAC
        value = int(read_config.get("borders_foreward", "AAC_l"))
        self.__aac_l = value
        value = int(read_config.get("borders_foreward", "AAC_k"))
        self.__aac_k = value
        #LOG
        value = int(read_config.get("borders_foreward", "LOG_l"))
        self.__log_l = value
        value = int(read_config.get("borders_foreward", "LOG_k"))
        self.__log_k = value

        ### BI Variation ##
        # DifBI
        value = int(read_config.get("borders_foreward", "DifBI_l"))
        self.__dif_bi_for_l = value
        value = int(read_config.get("borders_foreward", "DifBI_k"))
        self.__dif_bi_for_k = value
        # SigBI
        value = int(read_config.get("borders_foreward", "SigBI_l"))
        self.__std_bi_for_l = value
        value = int(read_config.get("borders_foreward", "SigBI_k"))
        self.__std_bi_for_k = value

        ### EMGT Variation ##
        # DifEMGT
        value = int(read_config.get("borders_foreward", "DifEMGT_l"))
        self.__dif_emgt_for_l = value
        value = int(read_config.get("borders_foreward", "DifEMGT_k"))
        self.__dif_emgt_for_k = value
        # SigBI
        value = int(read_config.get("borders_foreward", "SigEMGT_l"))
        self.__std_emgt_for_l = value
        value = int(read_config.get("borders_foreward", "SigEMGT_k"))
        self.__std_emgt_for_k = value

        ### IndBI ##
        # IndMinBI
        value = int(read_config.get("borders_foreward", "IndMinBI_l"))
        self.__ind_min_bi_for_l = value
        value = int(read_config.get("borders_foreward", "IndMinBI_k"))
        self.__ind_min_bi_for_k = value
        # IndMaxBI
        value = int(read_config.get("borders_foreward", "IndMaxBI_l"))
        self.__ind_max_bi_for_l = value
        value = int(read_config.get("borders_foreward", "IndMaxBI_k"))
        self.__ind_max_bi_for_k = value

        ### IndEMGT ##
        # IndMinEMGT
        value = int(read_config.get("borders_foreward", "IndMinEMGT_l"))
        self.__ind_min_emgt__for_l = value
        value = int(read_config.get("borders_foreward", "IndMinEMGT_k"))
        self.__ind_min_emgt_for_k = value
        # IndMaxEMGT
        value = int(read_config.get("borders_foreward", "IndMaxEMGT_l"))
        self.__ind_max_emgt_for_l = value
        value = int(read_config.get("borders_foreward", "IndMaxEMGT_k"))
        self.__ind_max_emgt_for_k = value


        ##################################################################
        ## SAX Converter
        sample_bi = self.__bi_sax_sample_before + self.__bi_sax_sample_after
        value = int(read_config.get("sax", "bi_bin"))
        symbols_bi = int(sample_bi / value)
        alphabeth_size = int(read_config.get("sax", "alphabeth_size_bi"))
        self.__bi_sax_converter = SAXConverter(symbols_bi, alphabeth_size, False)

        sample_emg = self.__emg_sax_sample_before + self.__emg_sax_sample_after
        value = int(read_config.get("sax", "emg_bin"))
        symbols_emg = int(sample_emg / value)
        alphabeth_size = int(read_config.get("sax", "alphabeth_size_emg"))
        self.__emg_sax_converter = SAXConverter(symbols_emg, alphabeth_size, True)

        ###################################################################
        ## segmentation

        delta_bi = float(read_config.get("segmentation", "delta_bi"))
        cut_off_bi = float(read_config.get("segmentation", "cut_off_bi"))
        desired_sample_rate = float(read_config.get("segmentation", "desired_sample_rate"))
        asstdning_window = float(read_config.get("segmentation", "assigning_window"))
        remove_double_swallows = int(read_config.get("segmentation", "remove_double_swallows"))

        if remove_double_swallows == 1:
            remove_double_swallows = True
        else:
            remove_double_swallows = False

        self.__segmentation = Segmentation(self.__data_file, delta_bi, asstdning_window, cut_off_bi, remove_double_swallows, desired_sample_rate)

        ###################################################################
        ## filter
        self.__cut_off_bi = float(read_config.get("filter", "cut_off_bi"))


    def get_feature(self):
        feature = self.__calculate_feature()
        return feature

    def get_start_of_swallows(self):

        maxima = np.concatenate([self.__preceding_true_maxima, self.__preceding_false_maxima])
        return maxima

    def get_cleaned_bi_100Hz(self):

        return self.__segmentation.getCleanedBiData100Hz()

    def getFeatureHeader(self):

        header = []
        bi_word_length = self.__bi_sax_converter.getWordLength()
        for i in range(0, bi_word_length):
            header.append("BI SAX " + str(i + 1))

        emg_word_length = self.__emg_sax_converter.getWordLength()
        for i in range(0, emg_word_length):
            header.append("EMG SAX " + str(i + 1))

        if self.__only_sax == False:
            header.append("DifBIr")
            header.append("DifBIf")
            header.append("SigBIr")
            header.append("SigBIf")
            header.append("IndMinBIr")
            header.append("IndMinBIf")
            header.append("IndMaxBIr")
            header.append("IndMaxBIf")
            header.append("DifEMGTr")
            header.append("DifEMGTf")
            header.append("SigEMGTr")
            header.append("SigEMGTf")
            header.append("RMS")
            header.append("ZCR")
            header.append("AAC")
            header.append("LOG")
            header.append("MaxEMG")
            header.append("IndMaxEMG1")
            header.append("IndMaxEMG2")
            header.append("IndMinEMG")


        header.append("Sample")
        header.append("Label")

        return header


    def __calculate_feature(self):

        true_features = []

        if self.__true_points.shape[0] > 0:
            for i in range(0, self.__true_points.shape[0]):
                features_help = self.__calc_feautre_vector(self.__true_points[i], self.__preceding_true_maxima[i])

                if features_help.shape[0] > 0:
                    true_features.append(np.concatenate((features_help, [self.__true_points[i], 1], self.__file_meta))) # add index and label


        false_features = []
        if self.__false_points.shape[0] > 0:
            for i in range(0, self.__false_points.shape[0]):

                features_help = self.__calc_feautre_vector(self.__false_points[i], self.__preceding_false_maxima[i])

                if features_help.shape[0] > 0:
                    false_features.append(np.concatenate((features_help, [self.__false_points[i], 0], self.__file_meta))) # add index and label

        if len(true_features) > 0 and len(false_features) > 0:
            feature = np.concatenate((np.asarray(true_features), np.asarray(false_features)))
        elif len(true_features) == 0 and len(false_features) > 0:
            feature = np.asarray(false_features)
        elif len(true_features) > 0 and len(false_features) == 0:
            feature = np.asarray(true_features)
        else:
            print("No Features in File")
            return np.zeros((0,))

        sample_index = -2-len(self.__file_meta)
        ind = np.argsort(feature[:, sample_index]) # sort feature vectors in order of appearance in data
        feature = feature[ind,:]
        return feature


    def __calc_feautre_vector(self, point_down, point_up):

        start_bi = point_down - self.__bi_sax_sample_before
        stop_bi = point_down + self.__bi_sax_sample_after

        point_down_4000 = int(point_down * 40)
        start_emg = point_down_4000 - self.__emg_sax_sample_before
        stop_emg = point_down_4000 + self.__emg_sax_sample_after



        #  check für data parts index needed !!
        check_start = start_bi > 0 and start_emg > 0 and \
                      point_down_4000 - self.__rms_l > 0 and \
                      point_down_4000 - self.__zcr_l > 0 and \
                      point_down_4000 - self.__aac_l > 0 and \
                      point_down_4000 - self.__log_l > 0 and \
                      point_down - self.__dif_bi_rev_l > 0 and \
                      point_down - self.__std_bi_rev_l > 0 and \
                      point_down_4000 - self.__std_emgt_rev_l > 0 and \
                      point_down_4000 - self.__ind_min_bi_rev_l > 0 and \
                      point_down_4000 - self.__ind_min_emgt_rev_l > 0 and \
                      point_down_4000 - self.__ind_max_emgt_rev_l > 0 and \
                      point_down_4000 - self.__max_emg_rev_l > 0 and \
                      point_down_4000 - self.__ind_max_emg_1_rev_l > 0 and \
                      point_down_4000 - self.__ind_max_emg_2_rev_l > 0 and \
                      point_down_4000 - self.__ind_min_emg_rev_l > 0 and \
                      point_down - self.__dif_bi_for_l > 0 and \
                      point_down - self.__std_bi_for_l > 0 and \
                      point_down_4000 - self.__std_emgt_for_l > 0 and \
                      point_down_4000 - self.__ind_min_bi_for_l > 0

        check_stop = stop_bi < self.__bi_data.shape[0] and \
                     stop_emg < self.__emg_data.shape[0] and \
                      point_down_4000 + self.__rms_k < self.__emg_data.shape[0] and \
                      point_down_4000 + self.__zcr_k < self.__emg_data.shape[0] and \
                      point_down_4000 + self.__aac_k < self.__emg_data.shape[0] and \
                      point_down_4000 + self.__log_k < self.__emg_data.shape[0] and \
                      point_down + self.__dif_bi_rev_k < self.__bi_data.shape[0] and \
                      point_down + self.__std_bi_rev_k < self.__bi_data.shape[0] and \
                      point_down_4000 + self.__std_emgt_rev_k < self.__emgt_data.shape[0] and \
                      point_down_4000 + self.__ind_min_bi_rev_k < self.__emgt_data.shape[0] and \
                      point_down_4000 + self.__ind_min_emgt_rev_k < self.__emgt_data.shape[0] and \
                      point_down_4000 + self.__ind_max_emgt_rev_k < self.__emgt_data.shape[0] and \
                      point_down_4000 + self.__max_emg_rev_k < self.__emg_data.shape[0] and \
                      point_down_4000 + self.__ind_max_emg_1_rev_k < self.__emg_data.shape[0] and \
                      point_down_4000 + self.__ind_max_emg_2_rev_k < self.__emg_data.shape[0] and \
                      point_down_4000 + self.__ind_min_emg_rev_k < self.__emg_data.shape[0] and \
                      point_down + self.__dif_bi_for_k < self.__bi_data.shape[0] and \
                      point_down + self.__std_bi_for_k < self.__bi_data.shape[0] and \
                      point_down_4000 + self.__std_emgt_for_k < self.__emgt_data.shape[0] and \
                      point_down_4000 + self.__ind_min_bi_for_k < self.__emgt_data.shape[0]


        features = np.zeros((0,))

        if check_start and check_stop:

            if self.__only_sax == False:

                ############
                # reverse
                # BI
                std_bi_rev = np.std(self.__bi_data[point_down-self.__std_bi_rev_l:point_down + self.__std_bi_rev_k])
                dif_bi_rev = np.max(self.__bi_data[point_down-self.__dif_bi_rev_l:point_down + self.__dif_bi_rev_k]) - np.min(self.__bi_data[point_down-self.__dif_bi_rev_l:point_down + self.__dif_bi_rev_k])
                ind_max_bi_rev = self.__getIndMaxMeanBin(self.__bi_data[point_down - self.__ind_max_bi_rev_l:point_down + self.__ind_max_bi_rev_k], 25)
                ind_min_bi_rev = self.__getIndMinMeanBin(self.__bi_data[point_down - self.__ind_min_bi_rev_l:point_down + self.__ind_min_bi_rev_k], 25)

                #EMGT
                std_emgt_rev = np.std(self.__emgt_data[point_down_4000 - self.__std_emgt_rev_l:point_down_4000 + self.__std_emgt_rev_k])
                dif_emgt_rev = np.max(self.__emgt_data[point_down_4000 - self.__dif_emgt_rev_l:point_down_4000 + self.__dif_emgt_rev_k]) - np.min(self.__emgt_data[point_down_4000-self.__dif_emgt_rev_l:point_down_4000 + self.__dif_emgt_rev_k])

                max_emg = self.__getMaxStdBin(self.__emg_data[point_down_4000 - self.__max_emg_rev_l:point_down_4000 + self.__max_emg_rev_k], 600)
                ind_max_emg_1 = self.__getIndMaxStdBin(self.__emg_data[point_down_4000 - self.__ind_max_emg_1_rev_l:point_down_4000 + self.__ind_max_emg_1_rev_k], 600)
                ind_max_emg_2 = self.__getIndMaxStdBin(self.__emg_data[point_down_4000 - self.__ind_max_emg_2_rev_l:point_down_4000 + self.__ind_max_emg_2_rev_k], 600)
                ind_min_emg = self.__getIndMinStdBin(self.__emg_data[point_down_4000 - self.__ind_min_emg_rev_l:point_down_4000 + self.__ind_min_emg_rev_k], 600)


                # foreward
                # BI
                std_bi_for = np.std(self.__bi_data[point_down - self.__std_bi_for_l:point_down + self.__std_bi_for_k])
                dif_bi_for = np.max(self.__bi_data[point_down - self.__dif_bi_for_l:point_down + self.__dif_bi_for_k]) - np.min(self.__bi_data[point_down-self.__dif_bi_for_l:point_down + self.__dif_bi_for_k])
                ind_max_bi_for = self.__getIndMaxMeanBin(self.__bi_data[point_down - self.__ind_max_bi_for_l:point_down +  self.__ind_max_bi_for_k], 25)
                ind_min_bi_for = self.__getIndMinMeanBin(self.__bi_data[point_down - self.__ind_min_bi_for_l:point_down + self.__ind_min_bi_for_k], 25)

                #EMGT
                std_emgt_for = np.std(self.__emgt_data[point_down_4000 - self.__std_emgt_for_l:point_down_4000 + self.__std_emgt_for_k])
                dif_emgt_for = np.max(self.__emgt_data[point_down_4000 - self.__dif_emgt_for_l:point_down_4000 + self.__dif_emgt_for_k]) - np.min(self.__emgt_data[point_down_4000 - self.__dif_emgt_for_l:point_down_4000 + self.__dif_emgt_for_k])

                # EMG
                rms = self.__calcRMSValue(self.__emg_data[point_down_4000 - self.__rms_l:point_down_4000 + self.__rms_k])
                zc = self.__calcZeroCrossingsValue(self.__emg_data[point_down_4000 - self.__zcr_l:point_down_4000 + self.__zcr_k])
                log = self.__calcLogEstimatorValue(self.__emg_data[point_down_4000 - self.__log_l:point_down_4000 + self.__log_k])
                aac = self.__calcAverageAmplitudeChange(self.__emg_data[point_down_4000 - self.__aac_l:point_down_4000 + self.__aac_k])

            # SAX
            emg_sax = self.__emg_sax_converter.convert(self.__emg_data[start_emg:stop_emg])
            bi_sax = self.__bi_sax_converter.convert(self.__bi_data[start_bi:stop_bi])


            if self.__only_sax:
                features = np.concatenate((bi_sax,emg_sax))

            else:
                features = np.concatenate((bi_sax,emg_sax,
                                           [std_bi_rev, std_bi_for, dif_bi_rev, dif_bi_for],
                                           [ind_min_bi_rev, ind_min_bi_for, ind_max_bi_rev, ind_max_bi_for],
                                           [dif_emgt_rev, dif_emgt_for, std_emgt_rev, std_emgt_for],
                                           [rms,zc,log,aac],
                                           [max_emg, ind_max_emg_1, ind_max_emg_2, ind_min_emg]))

        #else:
        #    print("Data Border Exceeded for Index: " + str(point_down))

        return features


    def __getMaxStdBin(self, data_in, bin_length):

        var_bins = []
        bins = math.floor(data_in.shape[0] / bin_length)

        for i in range(0, bins):
            var_bins.append(np.std(data_in[i*bin_length:(i+1)*bin_length]))

        max_index = np.argmax(var_bins)
        max = var_bins[max_index]
        return max

    def __getIndMaxStdBin(self, data_in, bin_length):

        var_bins = []
        bins = math.floor(data_in.shape[0] / bin_length)

        for i in range(0, bins):
            var_bins.append(np.std(data_in[i*bin_length:(i+1)*bin_length]))

        max_index = np.argmax(var_bins)
        return max_index

    def __getIndMinStdBin(self, data_in, bin_length):

        var_bins = []
        bins = math.floor(data_in.shape[0] / bin_length)

        for i in range(0, bins):
            var_bins.append(np.std(data_in[i*bin_length:(i+1)*bin_length]))

        min_index = np.argmin(var_bins)
        return min_index

    def __getIndMaxMeanBin(self, data_in, bin_length):

        var_bins = []
        bins = math.floor(data_in.shape[0] / bin_length)

        for i in range(0, bins):
            var_bins.append(np.mean(data_in[i * bin_length:(i + 1) * bin_length]))

        max_index = np.argmax(var_bins)
        return max_index

    def __getIndMinMeanBin(self, data_in, bin_length):

        var_bins = []
        bins = math.floor(data_in.shape[0] / bin_length)

        for i in range(0, bins):
            var_bins.append(np.mean(data_in[i * bin_length:(i + 1) * bin_length]))

        min_index = np.argmin(var_bins)
        return min_index

    def __calcSlopeSignChange(self, data):

        value = 0
        for i in range(0, data.shape[0] - 1):

            if (data[i - 1] - data[i]) * (data[i] - data[i + 1]) < 0:
                value = value + 1

        value = value / data.shape[0]
        return value

    def __calcLogEstimatorValue(self, data):

        value = 0
        for i in range(0, data.shape[0]):

            if data[i] != 0:
                value = value + np.log(abs(data[i]))

        value = value / data.shape[0]
        return value

    def __calcRMSValue(self, data):

        value = np.sqrt(np.mean(np.square(data)))
        return value

    def __calcZeroCrossingsValue(self, data):

        value = 0
        for i in range(0, data.shape[0] - 1):

            if data[i + 1] * data[i] < 0:
                value = value + 1

        value = value / data.shape[0]
        return value

    def __calcAverageAmplitudeChange(self, data):

        value = 0
        for i in range(0, data.shape[0] - 1):
            value = abs(data[i + 1] - data[i])

        value = (value / data.shape[0])
        return value