from EM_init import *
from EM import *
from plotGMM import *
from GMR import *
from matplotlib import pyplot as plt
import numpy as np

class GMM_GMR(object):

    def __init__(self, numberOfStates):
        self.numbefOfStates = numberOfStates

    def fit(self, data):
        self.data = data
        Priors, Mu, Sigma = EM_init(data, self.numbefOfStates)
        self.Priors, self.Mu, self.Sigma, self.Pix = EM(data, Priors, Mu, Sigma)

    def predict(self, inputMat):
        nbVar, nbData = np.shape(self.data)
        self.expData = np.ndarray(shape=(nbVar, np.size(inputMat)))
        self.expData[0, :] = inputMat
        self.expData[1:nbVar, :], self.expSigma = GMR(self.Priors, self.Mu, self.Sigma, self.expData[0, :], 0,
                                                      np.arange(1, nbVar))

    def getPredictedMatrix(self):
        return self.expData

    def plot(self, xAxis = 0, yAxis = 1, plotType = "Clusters", ax = plt, dataColor = [0, 0.8, 0.7],
             clusterColor = [0, 0.8, 0], regressionColor = [0,0,0.8]):
        xlim = [self.data[xAxis,:].min() - (self.data[xAxis,:].max() - self.data[xAxis,:].min())*0.1, self.data[xAxis,:].max() + (self.data[xAxis,:].max() - self.data[xAxis,:].min())*0.1]
        ylim = [self.data[yAxis,:].min() - (self.data[yAxis,:].max() - self.data[yAxis,:].min())*0.1, self.data[yAxis,:].max() + (self.data[yAxis,:].max() - self.data[yAxis,:].min())*0.1]
        if plotType == "Data":
            ax.plot(self.data[xAxis,:], self.data[yAxis,:],'.', color=dataColor)
            plt.xlim(xlim)
            plt.ylim(ylim)
        elif plotType == "Clusters":
            rows = np.array([xAxis, yAxis])
            cols = np.arange(0, self.numbefOfStates, 1)
            plotGMM(self.Mu[np.ix_(rows, cols)], self.Sigma[np.ix_(rows, rows, cols)], [0, 0.8, 0], 1, ax)
            plt.xlim(xlim)
            plt.ylim(ylim)
        elif plotType == "Regression":
            rows = np.array([xAxis, yAxis])
            rows2 = np.array([yAxis - 1, yAxis - 1])
            cols = np.arange(0, self.expData.shape[1], 1)
            cols = cols.astype(int)
            plotGMM(self.expData[np.ix_(rows, cols)], self.expSigma[np.ix_(rows2, rows2, cols)], regressionColor, 2, ax)
            plt.xlim(xlim)
            plt.ylim(ylim)
        else:
            print "Invalid plot type.\nPossible choices are: Data, Clusters, Regression."