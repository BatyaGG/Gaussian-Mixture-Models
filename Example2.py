from GMM_GMR import GMM_GMR
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
    data = np.loadtxt("data2.txt", delimiter=',')

    gmm = GMM_GMR(4)
    gmm.fit(data)
    timeInput = np.linspace(1, np.max(data[0, :]), 300)
    gmm.predict(timeInput)

    fig = plt.figure()
    fig.suptitle("Axis 1 vs axis 0")

    ax1 = fig.add_subplot(221)
    plt.title("Data")
    gmm.plot(ax=ax1, plotType="Data")

    ax2 = fig.add_subplot(222)
    plt.title("Gaussian States")
    gmm.plot(ax=ax2, plotType="Clusters")

    ax3 = fig.add_subplot(223)
    plt.title("Regression")
    gmm.plot(ax=ax3, plotType="Regression")

    ax4 = fig.add_subplot(224)
    plt.title("Clusters + Regression")
    gmm.plot(ax=ax4, plotType="Clusters")
    gmm.plot(ax=ax4, plotType="Regression")

    fig = plt.figure()
    fig.suptitle("Axis 2 vs axis 1")

    ax1 = fig.add_subplot(221)
    plt.title("Data")
    gmm.plot(ax=ax1, plotType="Data", xAxis=1, yAxis=2)

    ax2 = fig.add_subplot(222)
    plt.title("Gaussian States")
    gmm.plot(ax=ax2, plotType="Clusters", xAxis=1, yAxis=2)

    ax3 = fig.add_subplot(223)
    plt.title("Regression")
    gmm.plot(ax=ax3, plotType="Regression", xAxis=1, yAxis=2)

    ax4 = fig.add_subplot(224)
    plt.title("Clusters + Regression")
    gmm.plot(ax=ax4, plotType="Clusters", xAxis=1, yAxis=2)
    gmm.plot(ax=ax4, plotType="Regression", xAxis=1, yAxis=2)

    predictedMatrix = gmm.getPredictedMatrix()
    print predictedMatrix

    plt.show()