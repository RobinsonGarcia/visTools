#!python
import numpy as np
import math
cimport numpy as np
import GaussianFilter as cpy_GF
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
    G[i,:]=cpy_GF.GaussianFilter(3,3,1.5*xk[i,3]).flatten()
  #if x.max()>255.0:
  #  x/=255.0

  #cdef np.ndarray[np.float64_t,ndim=1] bins = np.linspace(0,1,n+1) + 1/(n+1)

  for ii in range(nn):
    for i in x[ii,:]:
      for j in range(n+1):
        if i < bins[j]:
          hist[ii,j]+=w[ii,j]*G[ii,i]
          break

  return hist
