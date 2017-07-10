from EM_init_kmeans import *
from EM import *
from plotGMM import *
from matplotlib import pyplot as plt
import numpy as np

class GMM(object):

    def __init__(self, numberOfStates):
        self.numbefOfStates = numberOfStates

    def fit(self, data):
        self.data = data
        Priors, Mu, Sigma = EM_init_kmeans(data, self.numbefOfStates)
        self.Priors, self.Mu, self.Sigma, self.Pix = EM(data, Priors, Mu, Sigma)

    def plot(self, xAxis = 0, yAxis = 1, plotType = "Data and clusters", ax = plt, dataColor = [0, 0.8, 0.7],
             clusterColor = [0, 0.8, 0]):
        if plotType == "Only data":
            ax.plot(self.data[xAxis,:], self.data[yAxis,:],'.', color=dataColor)
        elif plotType == "Only clusters":
            plotGMM(self.Mu, self.Sigma, clusterColor, 1, ax)
        elif plotType == "Data and clusters":
            ax.plot(self.data[xAxis, :], self.data[yAxis, :], '.', color=dataColor)
            plotGMM(self.Mu, self.Sigma, clusterColor, 1, ax)
        else:
            print "Invalid plot type.\nPossible choices are: Only data, Only clusters, Data and clusters."

if __name__ == "__main__":
    data = np.loadtxt("data.txt", delimiter=',')
    data = data[:, 0:2].T
    gmm = GMM(4)
    gmm.fit(data)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    gmm.plot(ax=ax)
    plt.show()