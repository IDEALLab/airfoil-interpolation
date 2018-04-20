"""
B-spline approximation.

Author(s): Wei Chen (wchen459@umd.edu)

X : Each row contains the coordinates of B-spline control points and 
    the parameter at the leading edge: 
        `[x0, ..., xn; y0, ..., yn; u_head]`.
"""

import os
import glob
import numpy as np
from matplotlib import pyplot as plt
from cartesian import read_cartesian
from interp import interpolate


N = 64 # number of interpolated points
k = 3 # degree of B-splines

data_paths = glob.glob("./data/*.*")
X = np.zeros((len(data_paths), N, 2))
fps = {}
iers = {}
for (i, data_path) in enumerate(data_paths):
    
    name = os.path.splitext(os.path.basename(data_path))[0]
    print i, name
    
    Q = read_cartesian(data_path)
    x_new, y_new, fp, ier = interpolate(Q, N, k)
    X[i] = np.vstack((x_new, y_new)).T
    fps[name] = fp
    iers[name] = ier
    
    plt.figure(figsize=(15, 10))
    plt.plot(Q[:,0], Q[:,1], 'ro', alpha=.5)
    plt.plot(x_new, y_new, 'bo-', alpha=.5)
    plt.axis('equal')
    plt.xlim(-0.1, 1.1)
    plt.title('%s  residuals: %f  %d' % (name, fp, ier))
    image_save_path = 'plots_interp/' + name + '.svg'
    plt.savefig(image_save_path)
    plt.close()
    
np.save('airfoil_interp.npy', X)
np.save('fps.npy', fps) # use np.load('fps.npy').item() to load the dictionary
np.save('iers.npy', iers) # use np.load('iers.npy').item() to load the dictionary