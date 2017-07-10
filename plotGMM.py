import numpy as np
from matplotlib.path import Path
import matplotlib.patches as patches
import scipy.linalg as lin
import matplotlib.pyplot as plt

def plotGMM(Mu, Sigma, color,display_mode, ax):
    a, nbData = np.shape(Mu)
    lightcolor = np.asarray(color) + np.asarray([0.6,0.6,0.6])
    a = np.nonzero(lightcolor > 1)
    lightcolor[a] = 1

    minsx = []
    maxsx = []
    minsy = []
    maxsy = []

    if display_mode==1:
        nbDrawingSeg = 40
        t = np.linspace(-np.pi,np.pi,nbDrawingSeg)
        t = np.transpose(t)
        for j in range (0,nbData):
            stdev = lin.sqrtm(3*Sigma[:,:,j])
            X = np.dot(np.transpose([np.cos(t), np.sin(t)]), np.real(stdev))
            X = X + np.tile(np.transpose(Mu[:,j]), (nbDrawingSeg,1))

            minsx.append(min(X[:,0]))
            maxsx.append(max(X[:,0]))
            minsy.append(min(X[:,1]))
            maxsy.append(max(X[:,1]))

            verts = []
            codes = []
            for i in range (0, nbDrawingSeg+1):
                if i==0:
                    vert = (X[0,0], X[0,1])
                    code = Path.MOVETO
                elif i!=nbDrawingSeg:
                    vert = (X[i,0], X[i,1])
                    code = Path.CURVE3
                else:
                    vert = (X[0,0], X[0,1])
                    code = Path.CURVE3
                verts.append(vert)
                codes.append(code)
            path = Path(verts, codes)
            patch = patches.PathPatch(path, facecolor=lightcolor, edgecolor=color, lw=2)
            ax.add_patch(patch)
            ax.plot(Mu[0,:], Mu[1,:], "x",color = color)
        # ax.set_xlim(min(minsx),max(maxsx))
        # ax.set_ylim(min(minsy),max(maxsy))
        plt.show()
    elif display_mode == 2:
        nbDrawingSeg = 40
        t = np.linspace(-np.pi, np.pi, nbDrawingSeg)
        t = np.transpose(t)

        for j in range(0, nbData):
            stdev = lin.sqrtm(3 * Sigma[:, :, j])
            X = np.dot(np.transpose([np.cos(t), np.sin(t)]), np.real(stdev))
            X = X + np.tile(np.transpose(Mu[:, j]), (nbDrawingSeg, 1))

            minsx.append(min(X[:, 0]))
            maxsx.append(max(X[:, 0]))
            minsy.append(min(X[:, 1]))
            maxsy.append(max(X[:, 1]))

            verts = []
            codes = []
            for i in range(0, nbDrawingSeg+1):
                if i == 0:
                    vert = (X[0, 0], X[0, 1])
                    code = Path.MOVETO
                elif i != nbDrawingSeg:
                    vert = (X[i, 0], X[i, 1])
                    code = Path.CURVE3
                else:
                    vert = (X[0, 0], X[0, 1])
                    code = Path.CURVE3
                verts.append(vert)
                codes.append(code)
            path = Path(verts, codes)
            patch = patches.PathPatch(path, linestyle=None, color = lightcolor)
            ax.add_patch(patch)
        ax.plot(Mu[0, :], Mu[1, :], "-",lw = 3, color=color)
        # ax.set_xlim(min(minsx), max(maxsx))
        # ax.set_ylim(min(minsy), max(maxsy))
        plt.show()