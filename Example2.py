from GMM_GMR import GMM_GMR
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
    data = np.loadtxt("data2.txt", delimiter=',')

    gmr = GMM_GMR(4)
    gmr.fit(data)
    timeInput = np.linspace(1, np.max(data[0, :]), 300)
    gmr.predict(timeInput)

    fig = plt.figure()
    fig.suptitle("Axis 1 vs axis 0")

    ax1 = fig.add_subplot(221)
    plt.title("Data")
    gmr.plot(ax=ax1, plotType="Data")

    ax2 = fig.add_subplot(222)
    plt.title("Gaussian States")
    gmr.plot(ax=ax2, plotType="Clusters")

    ax3 = fig.add_subplot(223)
    plt.title("Regression")
    gmr.plot(ax=ax3, plotType="Regression")

    ax4 = fig.add_subplot(224)
    plt.title("Clusters + Regression")
    gmr.plot(ax=ax4, plotType="Clusters")
    gmr.plot(ax=ax4, plotType="Regression")

    fig = plt.figure()
    fig.suptitle("Axis 2 vs axis 1")

    ax1 = fig.add_subplot(221)
    plt.title("Data")
    gmr.plot(ax=ax1, plotType="Data", xAxis=1, yAxis=2)

    ax2 = fig.add_subplot(222)
    plt.title("Gaussian States")
    gmr.plot(ax=ax2, plotType="Clusters", xAxis=1, yAxis=2)

    ax3 = fig.add_subplot(223)
    plt.title("Regression")
    gmr.plot(ax=ax3, plotType="Regression", xAxis=1, yAxis=2)

    ax4 = fig.add_subplot(224)
    plt.title("Clusters + Regression")
    gmr.plot(ax=ax4, plotType="Clusters", xAxis=1, yAxis=2)
    gmr.plot(ax=ax4, plotType="Regression", xAxis=1, yAxis=2)

    predictedMatrix = gmr.getPredictedMatrix()
    print predictedMatrix

    plt.show()