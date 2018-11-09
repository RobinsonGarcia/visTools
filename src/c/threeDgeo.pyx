import numpy as np
import math
cimport numpy as np

DTYPE = np.int
#ctypedef np.int_t DTYPE_t


cpdef homography_error(pairs,h):

    H = np.reshape(h,(3,3))

    x0 = np.vstack((pairs[0][:,:2].T,np.ones(pairs[0].shape[0])))

    x1_ = H.dot(x0)

    x1_ = np.divide(x1_[:2,:],x1_[2,:],out=np.zeros_like(x1_[:2,:]),
                    where=x1_[2,:]!=0)

    return np.sum((x1_ - pairs[1][:,:2].T)**2,axis=0)
