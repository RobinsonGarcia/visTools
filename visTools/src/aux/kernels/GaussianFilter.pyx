import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef GaussianFilter(int w,int h,float sigma):
    cdef float E = 2.718281828459045
    cdef int m = int((w-1)/2)
    cdef int n = int((h-1)/2)
    cdef np.ndarray[np.float64_t,ndim=2] G = np.empty((h,w))
    cdef double Gsum = 0.0
    cdef int i,j
    for i in range(w):
        for j in range(h):
            G[j,i] = E**(-1*((i-m)**2+(j-n)**2)/(2*sigma**2))
            Gsum += G[j,i]

    return G/Gsum
