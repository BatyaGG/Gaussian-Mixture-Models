import numpy as np
import sys
from gaussPDF import gaussPDF
def GMR(Priors, Mu, Sigma, x, input, output):
    lo = np.size(output)
    nbData = np.size(x)
    nbVar = np.size(Mu, 0)
    nbStates = np.size(Sigma, 2)
    realmin = sys.float_info[3]
    Pxi = np.ndarray(shape=(nbData, nbStates))
    x = np.reshape(x,(1,nbData))
    y_tmp = np.ndarray(shape = (nbVar-1, nbData, nbStates))
    Sigma_y_tmp = np.ndarray(shape = (lo, lo, 1, nbStates))

    for i in range (0,nbStates):
        m = Mu[input,i]
        m = np.reshape(m,(np.size(input),1))
        s = Sigma[input, input, i]
        s = np.reshape(s, (np.size(input),np.size(input)))
        Pxi[:,i] = np.multiply(Priors[i],gaussPDF(x,m,s))
    beta = np.divide(Pxi,np.tile(np.reshape(np.sum(Pxi,1),(nbData, 1))+realmin,(1,nbStates)))
    for j in range (0,nbStates):
        a = np.delete(Mu, np.s_[input], axis = 0)
        a = a[:,j]
        a = np.reshape(a,(nbVar-np.size(input),1))
        a = np.tile(a, (1, nbData))
        b = np.delete(Sigma[:,:,j], 0, axis = 0)
        b = np.delete(b, np.s_[1:nbVar], axis = 1)
        c = Sigma[input, input, j]
        c = np.reshape(c, (1,1))
        c = np.linalg.inv(c)
        c = np.dot(b, c)
        d = np.reshape(Mu[input, j], (1,1))
        d = np.tile(d, (1,nbData))
        d = x - d
        d = np.dot(c, d)
        y_tmp[:,:,j] = a + d
    # pravilno
    a, b = np.shape(beta)
    beta_tmp = np.reshape(beta, (1,a,b))
    a = np.tile(beta_tmp,(lo,1,1))
    y_tmp2 = a*y_tmp
    # print('1')
    # print(y_tmp2[:,:,0])
    # print('2')
    # print(y_tmp2[:, :, 1])
    # print('3')
    # print(y_tmp2[:, :, 2])
    # print('4')
    # print(y_tmp2[:, :, 3])
    y = np.sum(y_tmp2,2)

    for j in range(0, nbStates):
        a = np.delete(Sigma[:,:,j], 0, axis = 0)
        a = np.delete(a, 0, axis = 1)
        b = np.delete(Sigma[:, :, j], 0, axis=0)
        b = np.delete(b, np.s_[1:nbVar], axis=1)
        c = Sigma[input, input, j]
        c = np.reshape(c, (1, 1))
        c = np.linalg.inv(c)
        c = np.dot(b, c)
        d = np.delete(Sigma[:,:,j], 0,axis = 1)
        d = np.delete(d, np.s_[1:nbVar], axis = 0)
        d = np.dot(c, d)
        Sigma_y_tmp[:,:,0,j] = a - d
    a, b = np.shape(beta)
    beta_tmp = np.reshape(beta,(1,1,a,b))
    a = beta_tmp*beta_tmp
    a = np.tile(a, (lo,lo,1,1))
    b = np.tile(Sigma_y_tmp,(1,1,nbData,1))
    Sigma_y_tmp2 = a*b
    Sigma_y = np.sum(Sigma_y_tmp2, 3)
    return (y, Sigma_y)