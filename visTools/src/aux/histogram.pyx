from visTools.src.aux.kernels.GaussianFilter import GaussianFilter
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef histogram(np.ndarray[np.float64_t,ndim=2] x,
np.ndarray[np.float64_t,ndim=2] w,
np.ndarray[np.float64_t,ndim=1] bins,
np.ndarray[np.float64_t,ndim=2] xk,int n):

    cdef int nn = x.shape[0]

    cdef np.ndarray[np.float64_t,ndim=2] hist = np.zeros((nn,n+1))

    cdef np.ndarray[np.float64_t,ndim=2] G = np.zeros((nn,9))

    cdef int i,ii,j

    for i in range(nn):
        G[i,:] = GaussianFilter(3,3,1.5*xk[i,3]).flatten()


    for ii in range(nn):
        for i in x[ii,:]:
            for j in range(n+1):
                if i < bins[j]:
                    hist[ii,j]+=w[ii,j]*G[ii,i]
                    break

    return hist
