# cython: cdivision=True
# cython: initializedcheck=False

import numpy as np
cimport numpy as np
from cython cimport wraparound, boundscheck, cdivision
from libc.math cimport fabs, sqrt, INFINITY, nearbyint
np.import_array()

cpdef int sor(double[:,:] a, double[:,:] b, double[:,:] c, double[:,:] d,
              double[:,:] e, double[:,:] f, double[:,:] u, double rho_jac):
    """Successive Over-Relaxation with Chebyshev acceleration."""
    cdef:
        int max_iter = 1000
        double eps = 1e-13
        double ac_norm_f = 0.0, omega = 1.0
        double resid, ac_norm_resid
        int nrows = a.shape[0], ncols = a.shape[1]
        int i, j, i_start, j_start, color, n

    for i in range(1, nrows-1):
        for j in range(1, ncols-1):
            ac_norm_f += fabs(f[i,j])

    for n in range(0, max_iter):
        ac_norm_resid = 0.0
        i_start = 1
        for color in range(0, 2):
            j_start = i_start
            for i in range(1, nrows-1, 2):
                for j in range(j_start, ncols-1, 2):
                    resid = (a[i,j]*u[i+1,j] + b[i,j]*u[i-1,j] + c[i,j]*u[i,j+1] 
                             + d[i,j]*u[i,j-1] + e[i,j]*u[i,j] - f[i,j])
                    ac_norm_resid += fabs(resid)
                    u[i,j] -= omega * resid / e[i,j]
                j_start = 3 - j_start
            i_start = 3 - i_start

            omega = (1.0 / (1.0 - 0.5*rho_jac*rho_jac)
                     if n == 0 and color == 0
                     else 1.0 / (1.0 - 0.25*rho_jac*rho_jac*omega))

        if ac_norm_resid < eps*ac_norm_f:
            return n
