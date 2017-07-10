import numpy as np
import math
import sys
def gaussPDF(Data, Mu, Sigma):
    realmin = sys.float_info[3]
    nbVar, nbData = np.shape(Data)
    Data = np.transpose(Data) - np.tile(np.transpose(Mu), (nbData, 1))
    prob = np.sum(np.dot(Data, np.linalg.inv(Sigma))*Data, 1)
    prob = np.exp(-0.5*prob)/np.sqrt((np.power((2*math.pi), nbVar))*np.absolute(np.linalg.det(Sigma))+realmin)
    return prob