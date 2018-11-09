import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef get_patches3(np.ndarray[np.float64_t,ndim=3] D, np.ndarray[np.float_t,ndim=2] xk, int h=3, int w=3, stride=1):

    cdef int N = D.shape[0]
    cdef int H = D.shape[1]
    cdef int W = D.shape[2]


    cdef int h_pad = int((H * (stride - 1) - stride + h) / 2)
    cdef int w_pad = int((W * (stride - 1) - stride + w) / 2)

    cdef int h_ = int((h - 1) / 2)
    cdef int w_ = int((w - 1) / 2)
    cdef int k,i,j
    cdef int n = xk.shape[0]

    cdef np.ndarray[np.float64_t,ndim=3] patches = np.zeros((n,h,w))
    cdef np.ndarray[np.float64_t,ndim=3] D_padded = np.pad(D, ((0, 0), (h_pad, h_pad), (w_pad, w_pad)), 'constant')

    for f in range(n):
        i, j,k,_ = xk[f].astype(int)

        i += h_pad
        j += w_pad
        patches[f,:,:] = D_padded[k, i - h_:i + h_ + 1, j - w_:j + w_ + 1]


    return patches
