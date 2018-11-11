import numpy as np
import math
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef GaussianFilter(int w,int h,float sigma):
    cdef int m = (w-1)/2
    cdef int n = (h-1)/2
    cdef np.ndarray[np.float64_t,ndim=2] G = np.empty((h,w))
    cdef double Gsum = 0.0
    cdef int i,j
    for i in range(w):
        for j in range(h):
            G[j,i] = math.e**(-1*((i-m)**2+(j-n)**2)/(2*sigma**2))
            Gsum += G[j,i]

    return G/Gsum
