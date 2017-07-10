import numpy as np
from scipy.cluster.vq import kmeans,vq
def EM_init_kmeans(Data, nbStates):
    nbVar, nbData = np.shape(Data)
    Priors = np.ndarray(shape = (1, nbStates))
    Sigma = np.ndarray(shape = (nbVar, nbVar, nbStates))
    Centers, _ = kmeans(np.transpose(Data), nbStates)
    Data_id, _ = vq(np.transpose(Data), Centers)
    Mu = np.transpose(Centers)
    for i in range (0,nbStates):
        idtmp = np.nonzero(Data_id==i)
        idtmp = list(idtmp)
        idtmp = np.reshape(idtmp,(np.size(idtmp)))
        Priors[0,i] = np.size(idtmp)
        a = np.concatenate((Data[:, idtmp],Data[:, idtmp]), axis = 1)
        Sigma[:,:,i] = np.cov(a)
        Sigma[:,:,i] = Sigma[:,:,i] + 0.00001 * np.diag(np.diag(np.ones((nbVar,nbVar))))
    Priors = Priors / nbData
    return (Priors, Mu, Sigma)
