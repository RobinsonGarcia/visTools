import numpy as np
import GaussianFilter as cpy_GF
import math
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef get_descriptors(np.ndarray[np.float64_t,ndim=3] Dmod,np.ndarray[np.float64_t,ndim=3] Drho,np.ndarray[np.float64_t,ndim=2] xk,int block_size=16,int cell_size=4,int nBins=8):

    cdef np.ndarray[np.float64_t,ndim=1] sigmas = np.unique(xk[:,3])
    cdef int nSigmas = sigmas.shape[0]

    #cdef np.ndarray[np.float64_t,ndim=3] weights = np.zeros((nSigmas,block_size,block_size))

    cdef int st = 0
    cdef float sig

    weights = {}

    for sig in sigmas:
        weights[np.round(sig,2)] = cpy_GF.GaussianFilter(block_size,block_size,sig)


    cdef int nImages = Dmod.shape[0] #number of images

    cdef int nKps = xk.shape[0] #number of keypoints

    cdef int half_block = int(block_size/2)

    cdef int nCells = int((block_size/cell_size)**2)

    cdef int half_cell = int(cell_size/2)

    cdef np.ndarray[np.float64_t,ndim=5] block = np.zeros((nImages,nKps,nCells,cell_size+1,cell_size+1))

    cdef np.ndarray[np.float64_t,ndim=3] Dmod_padded = np.pad(Dmod,((0,0),(half_block,half_block),(half_block,half_block)),'linear_ramp')
    cdef np.ndarray[np.float64_t,ndim=3] Drho_padded = np.pad(Drho,((0,0),(half_block,half_block),(half_block,half_block)),'linear_ramp')

    cdef np.ndarray[np.int_t,ndim=1] HH = np.arange(-half_block,half_block,cell_size).astype(int)
    cdef np.ndarray[np.int_t,ndim=1] WW = np.arange(-half_block,half_block,cell_size).astype(int)

    cdef np.ndarray[np.int_t,ndim=1] hh = np.arange(0,cell_size+1).astype(int)
    cdef np.ndarray[np.int_t,ndim=1] ww = np.arange(0,cell_size+1).astype(int)

    cdef np.ndarray[np.float64_t,ndim=1] bins = np.linspace(math.pi/nBins,2*math.pi,nBins)
    cdef np.ndarray[np.float64_t,ndim=2] histogram = np.zeros((nKps,nBins*nCells))

    cdef np.ndarray[np.float64_t,ndim=1] kp

    cdef np.ndarray[np.float64_t,ndim=1] mod

    cdef int tt,t0,t1,t2,t3,t4,scale_idx,iii,jjj,ii,jj,i,j
    cdef double rho_ij,rho,scale_sigma,o

    xk[:,3] = np.round(xk[:,3],2)

    tt=0
    for im in range(nImages):
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
                            rho_ij = Drho_padded[scale_idx,half_block+iii+i+ii,half_block+jjj+j+jj]-rho
                            t4 = 0
                            for o in bins:
                                if rho_ij<o:
                                    Dmod_ij = Dmod_padded[scale_idx,half_block+iii+i+ii,half_block+jjj+j+jj]
                                    histogram[t0,t4+t1*nBins]+= Dmod_ij*weights[scale_sigma][ii+i,jj+j]
                                t4+=1
                            block[tt,t0,t1,t2,t3] = Dmod_padded[scale_idx,half_block+iii+i+ii,half_block+jjj+j+jj]
                            t3+=1
                        t2+=1
                    t1+=1
            t0+=1
        tt+=1

    mod = np.sqrt(np.sum(histogram,axis=1))
    histogram = np.divide(histogram,mod[:,np.newaxis],out=np.zeros_like(histogram),where=mod[:,np.newaxis]!=0)
    histogram = np.clip(histogram,a_min=0,a_max=0.2)
    mod = np.sqrt(np.sum(histogram,axis=1))
    histogram = np.divide(histogram,mod[:,np.newaxis],out=np.zeros_like(histogram),where=mod[:,np.newaxis]!=0)

    return block,histogram
