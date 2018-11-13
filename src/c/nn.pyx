import numpy as np
import math
cimport numpy as np

DTYPE = np.int
#ctypedef np.int_t DTYPE_t

cpdef nearest_neighboor(np.ndarray[np.float64_t,ndim=2] fk1,
np.ndarray[np.float64_t,ndim=2] xk1,
np.ndarray[np.float64_t,ndim=2] fk2,
np.ndarray[np.float64_t,ndim=2] xk2,
double crt=0.8):
  '''
  cdef np.ndarray[np.float64_t,ndim=2] fk=fk1
  cdef int count=0
  cdef np.ndarray[np.int_t] idx2 = np.empty(0,dtype=np.int)
  cdef np.ndarray[np.int_t] idx1= np.empty(0,dtype=np.int)
  cdef np.ndarray[np.float64_t,ndim=1] phi_ = np.empty(0,dtype=np.float64)

  cdef int f1
  cdef int f2
  cdef np.ndarray[np.float64_t] v
  cdef np.ndarray[np.float64_t] d
  cdef float phi
  cdef np.ndarray[np.int_t] idx
  '''

  cdef np.ndarray[np.float64_t,ndim=1] a_sq = np.sum(fk1**2,axis=1)

  cdef np.ndarray[np.float64_t,ndim=2] ab = np.matmul(fk1,fk2.T)

  cdef np.ndarray[np.float64_t,ndim=1] b_sq = np.sum(fk2**2,axis=1)

  cdef np.ndarray[np.float64_t,ndim=2] a_b = a_sq[:,np.newaxis]+b_sq

  cdef np.ndarray[np.float64_t,ndim=2] dists = np.sqrt(a_b - 2*ab).T

  cdef int n = fk2.shape[0]

  cdef np.ndarray[np.int_t,ndim=1] idx1 = np.zeros(0).astype(int)
  cdef np.ndarray[np.int_t,ndim=1] idx2 = np.zeros(0).astype(int)
  cdef np.ndarray[np.float64_t,ndim=1] phi = np.zeros(0)

  cdef np.ndarray[np.int_t,ndim=1] id

  cdef float phi_

  cdef int i

  for i in range(n):
    id = np.argsort(dists[i,:])[:2]


    phi_  = np.sum((-fk1[id[0]] + fk2[i])**2)/np.sum((-fk1[id[1]] + fk2[i])**2)
    if phi_ < crt:
      idx1 = np.append(idx1,id[0])
      idx2 = np.append(idx2,i)
      phi = np.append(phi,phi_)


  pairs = (xk1[idx1.astype(int),:],xk2[idx2.astype(int)])

  idx = np.argsort(phi)
  return pairs,idx
