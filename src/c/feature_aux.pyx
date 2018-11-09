import numpy as np
import math
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cdef GaussianFilter(int w,int h,float sigma):
    cdef int m = (w-1)/2
    cdef int n = (h-1)/2
    cdef np.ndarray[np.float64_t,ndim=2] G = np.empty((h,w))
    cdef double Gsum = 0.0
    for i in range(w):
        for j in range(h):
            G[j,i] = math.e**(-1*((i-m)**2+(j-n)**2)/(2*sigma**2))
            Gsum += G[j,i]

    return G/Gsum

cdef split_matrix(A):
    x = (np.tile(np.tile(np.arange(4),16) + np.repeat(np.arange(0,14,4),16),4).reshape((4,-1)).T).astype(int)
    y = (np.tile(np.repeat(np.arange(0,14,4),4),4)[:,np.newaxis]+np.arange(4)).astype(int)
    return np.dstack(np.split(A[x,y],16))


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef build_fk(int N, np.ndarray[np.float64_t,ndim=2] xk,np.ndarray[np.int_t,ndim=3] hist_patches,np.ndarray[np.float_t,ndim=3]Imag,double ilum_sat):

  cdef np.ndarray[np.float64_t,ndim=1] fk1
  cdef np.ndarray[np.float64_t,ndim=2] fk = np.zeros((N,8*16)).astype(np.float)
  cdef np.ndarray[np.float64_t,ndim=2] sig_
  cdef np.ndarray[np.float64_t,ndim=1] contrib
  cdef np.ndarray[np.float64_t,ndim=3] sig

  for kp in range(N):
      fk1=np.empty(0).astype(np.float)
      sig = split_matrix(GaussianFilter(16,16,1.5*xk[kp,3]))
      for i in range(16):
        sig_ = sig[:,:,i]
        contrib = np.zeros(8).astype(np.float)
        for j in range(16):
          contrib[hist_patches[j,i,kp]]+=Imag[j,i,kp]*sig_.flatten()[j]

        fk1 = np.append(fk1,contrib)

      fk1 = fk1/np.linalg.norm(fk1)
      fk1 = np.clip(fk1,None,ilum_sat)
      fk[kp,:] = fk1/np.linalg.norm(fk1)

  return fk
