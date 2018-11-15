import numpy as np
import GaussianFilter as cpy_GF
import math
cimport numpy as np
cimport cython

'''
xks:
    0 - x
    1 - y
    2 - scale index
    3 - scale/sigma
    4 - grad orientations
'''
@cython.boundscheck(False)
@cython.wraparound(False)
cdef get_descriptors_idx(np.ndarray[np.float64_t,ndim=2] xk,int block_size=16,int cell_size=4,int nBins=8):


    cdef int nKps = xk.shape[0] #number of keypoints

    cdef int half_block = int(block_size/2)

    cdef int nCells = int((block_size/cell_size)**2)

    cdef int half_cell = int(cell_size/2)

    cdef np.ndarray[np.int_t,ndim=4] block_s = np.zeros((nKps,nCells,cell_size+1,cell_size+1)).astype(np.int)
    cdef np.ndarray[np.int_t,ndim=4] block_i = np.zeros((nKps,nCells,cell_size+1,cell_size+1)).astype(np.int)
    cdef np.ndarray[np.int_t,ndim=4] block_j = np.zeros((nKps,nCells,cell_size+1,cell_size+1)).astype(np.int)

    cdef np.ndarray[np.int_t,ndim=1] HH = np.arange(-half_block,half_block,cell_size)
    cdef np.ndarray[np.int_t,ndim=1] WW = np.arange(-half_block,half_block,cell_size)

    cdef np.ndarray[np.int_t,ndim=1] hh = np.arange(0,cell_size+1)
    cdef np.ndarray[np.int_t,ndim=1] ww = np.arange(0,cell_size+1)

    cdef int t0,scale_idx,iii,jjj,t1,ii,jj,t2,i,t3,j

    cdef double scale_sigma,rho

    t0=0
    for kp in xk:
        scale_idx = int(kp[2])
        scale_sigma = kp[3]
        rho = kp[4]
        iii,jjj = kp[:2].astype(int)
        t1=0
        for ii in HH:
            for jj in WW:
                t2 = 0
                for i in hh:
                    t3 = 0
                    for j in ww:

                        block_s[t0,t1,t2,t3] = scale_idx
                        block_i[t0,t1,t2,t3] = half_block+iii+i+ii
                        block_j[t0,t1,t2,t3] = half_block+jjj+j+jj

                        t3+=1
                    t2+=1

                t1+=1

        t0+=1

    return block_s,block_i,block_j



@cython.boundscheck(False)
@cython.wraparound(False)
cdef get_hist(np.ndarray[np.int_t,ndim=4] idx,np.ndarray[np.float64_t,ndim=3] Dmod,
np.ndarray[np.float64_t,ndim=2] xk,int block_size=16,int cell_size=4,int nBins=8):

    cdef np.ndarray[np.float64_t,ndim=1] sigmas = np.unique(xk[:,3])

    cdef int nSigmas = sigmas.shape[0]

    cdef np.ndarray[np.float64_t,ndim=2] histogram = np.zeros((idx.shape[0],nBins*idx.shape[1]))

    weights = {}
    for sig in sigmas:
        weights[sig] = cpy_GF.GaussianFilter(block_size,block_size,sig)

    cdef int nKps = xk.shape[0] #number of keypoints

    cdef int half_block = int(block_size/2)

    cdef int nCells = int((block_size/cell_size)**2)

    cdef int half_cell = int(cell_size/2)

    cdef np.ndarray[np.float64_t,ndim=3] Dmod_padded = np.pad(Dmod,((0,0),(half_block,half_block),(half_block,half_block)),'linear_ramp')

    cdef np.ndarray[np.int_t,ndim=1] HH = np.arange(-half_block,half_block,cell_size)
    cdef np.ndarray[np.int_t,ndim=1] WW = np.arange(-half_block,half_block,cell_size)

    cdef np.ndarray[np.int_t,ndim=1] hh = np.arange(0,cell_size+1)
    cdef np.ndarray[np.int_t,ndim=1] ww = np.arange(0,cell_size+1)

    cdef int t0,scale_idx,iii,jjj,t1,ii,t2,i,jj,t3,j
    cdef double scale_sigma,rho

    cdef float mod

    t0=0
    for kp in xk:
        scale_idx = int(kp[2])
        scale_sigma = kp[3]
        rho = kp[4]
        iii,jjj = kp[:2].astype(int)
        t1=0
        for ii in HH:
            for jj in WW:
                t2 = 0
                for i in hh:
                    t3 = 0
                    for j in ww:
                        mod = Dmod_padded[scale_idx,half_block+iii+i+ii,half_block+jjj+j+jj]
                        histogram[t0,idx[t0,t1,t2,t3]+t1*8]+= mod*weights[scale_sigma][ii+i,jj+j]
                        t3+=1
                    t2+=1
                t1+=1
        t0+=1

    return histogram


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef build_fk(np.ndarray[np.float64_t,ndim=3] Dmod,np.ndarray[np.float64_t,ndim=3] Drho,
np.ndarray[np.float64_t,ndim=2] xk,int block_size=16,int cell_size=4,int nBins=8):

    cdef np.ndarray[np.int_t,ndim=4] s,i,j

    s,i,j = get_descriptors_idx(xk,block_size,cell_size,nBins=nBins)

    cdef int half_block = int(block_size/2)

    cdef np.ndarray[np.float64_t,ndim=3] Dmod_padded = np.pad(Dmod,((0,0),(half_block,half_block),(half_block,half_block)),'linear_ramp')
    cdef np.ndarray[np.float64_t,ndim=3] Drho_padded = np.pad(Drho,((0,0),(half_block,half_block),(half_block,half_block)),'linear_ramp')


    cdef np.ndarray[np.float64_t,ndim=4] Dmod_patches = Dmod_padded[s,i,j]
    cdef np.ndarray[np.float64_t,ndim=4] Drho_patches = Drho_padded[s,i,j]

    cdef np.ndarray[np.float64_t,ndim=4] relRho = Drho_patches - xk[:,3][:,np.newaxis,np.newaxis,np.newaxis]

    relRho[relRho<0]+=2*math.pi

    cdef np.ndarray[np.float64_t,ndim=1] bins = np.linspace(math.pi/nBins,2*math.pi,nBins)

    cdef np.ndarray[np.int_t,ndim=4] idx = np.searchsorted(bins, relRho,'right')

    cdef np.ndarray[np.float64_t,ndim=2] histogram = get_hist(idx,Dmod,xk,block_size,cell_size,nBins)

    return histogram
