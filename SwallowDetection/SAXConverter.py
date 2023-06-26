import numpy as np
import scipy.signal
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from scipy.stats import chisquare

class SAXConverter:

    def __init__(self, word_length, alphabet_size, chi2):

        self.__word_length = word_length
        self.__alphabet_size = alphabet_size
        self.__chi2 = chi2

        self.__scaler = StandardScaler()

        if chi2:
            self.__cut_points = scipy.stats.chi2.ppf(np.linspace(1 / alphabet_size, 1, alphabet_size ), df=1)[0:-1]
            self.__cut_points = np.append(self.__cut_points, [math.inf])
        else:
            self.__cut_points = norm.ppf(np.linspace(0, 1, alphabet_size +1))[1:-1]
            self.__cut_points = np.append(self.__cut_points, [math.inf])

    def getScaledData(self):
        return self.__data_scaled

    def getCutPoints(self):
        return self.__cut_points

    def getWordLength(self):
        return self.__word_length

    def getAlphabetSize(self):
        return self.__alphabet_size

    def getPAA(self):
        return self.__paa

    def getSAX(self):
        return self.__sax

    def convert(self, data):

        self.__data_scaled = self.__scaler.fit_transform(data.reshape(-1, 1)).T[0]
        self.__paa = []
        self.__sax = []

        if self.__chi2:
            self.__data_scaled = np.square(self.__data_scaled)

        samples_per_symbol = int(self.__data_scaled.shape[0] / self.__word_length)

        for i in range(0, self.__word_length):

            if (i+1)*samples_per_symbol <= self.__data_scaled.shape[0]:
                self.__paa.append(np.mean(self.__data_scaled[i*samples_per_symbol:(i+1)*samples_per_symbol]))

        sax = []
        for p in self.__paa:
            symbol = 0
            for b in self.__cut_points:
                symbol = symbol + 1
                if p < b:
                    self.__sax.append(symbol)
                    break

        self.__sax = np.asarray(self.__sax)
        return self.__sax