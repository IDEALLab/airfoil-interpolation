"""
B-spline approximation.

Author(s): Wei Chen (wchen459@umd.edu)

Reference(s): 
    [1] Lepine, Jerome, Guibault, Francois, Trepanier, Jean-Yves, Pepin, Francois. (2001). 
        Optimized nonuniform rational B-spline geometrical representation for aerodynamic 
        design of wings. AIAA journal, 39(11), 2033-2041.
    [2] Lepine, J., Trepanier, J. Y., & Pepin, F. (2000, January). Wing aerodynamic design 
        using an optimized NURBS geometrical representation. In 38th Aerospace Sciences 
        Meeting and Exhibit (p. 669).

n+1 : number of control points
m+1 : number of data points
"""


import os
import glob
import numpy as np
from scipy.interpolate import splev, splprep, interp1d
from scipy.integrate import cumtrapz
from matplotlib import pyplot as plt
from cartesian import read_cartesian

def interpolate(Q, N, k, D=20, resolution=1000):
    ''' Interpolate N points whose concentration is based on curvature. '''
    res, fp, ier, msg = splprep(Q.T, u=None, k=k, s=1e-6, per=0, full_output=1)
    tck, u = res    
    uu = np.linspace(u.min(), u.max(), resolution)
    x, y = splev(uu, tck, der=0)
    dx, dy = splev(uu, tck, der=1)
    ddx, ddy = splev(uu, tck, der=2)
    cv = np.abs(ddx*dy - dx*ddy)/(dx*dx + dy*dy)**1.5 + D
    cv_int = cumtrapz(cv, uu, initial=0)
    fcv = interp1d(cv_int, uu)
    cv_int_samples = np.linspace(0, cv_int.max(), N)
    u_new = fcv(cv_int_samples)
    x_new, y_new = splev(u_new, tck, der=0)
    return x_new, y_new, fp, ier

if __name__ == "__main__":
    
    N = 64
    k = 3
    data_paths = glob.glob("./data/*.*")
    data_path = './data/2032c.dat'#data_paths[10]
    print data_path
    name = os.path.splitext(os.path.basename(data_path))[0]
    
    Q = read_cartesian(data_path)
    x_new, y_new, fp, ier = interpolate(Q, N, k)
    
    plt.figure()
    plt.plot(Q[:,0], Q[:,1], 'ro', alpha=.5)
    plt.plot(x_new, y_new, 'bo-', alpha=.5)
    plt.axis('equal')
    plt.xlim(-0.1, 1.1)
    plt.title('%s  residuals: %f  %d' % (name, fp, ier))
    plt.show()



    