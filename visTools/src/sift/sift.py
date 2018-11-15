
# coding: utf-8

# # Scale Invariant Feature Detector
# by Robinson Garcia
#
# sources:
#
#
# Ref: https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
# Ref: https://www.dropbox.com/sh/26xgy96py8itk14/AABhgCbYzraeSMkDpjY92kFVa/Lecture%20slides?dl=0&preview=2017f.Week10.Topic15.lecture.sift.pdf&subfolder_nav_tracking=1
# Ref: https://cs.nyu.edu/~fergus/teaching/vision_2012/3_Corners_Blobs_Descriptors.pdf
# Ref: https://arxiv.org/pdf/1603.09114.pdf

# In[1]:


'''python libraries'''
import numpy as np
import math
import matplotlib.pyplot as plt
import visTools.src.aux.support_functions as sf
from visTools.src.DoG import *
from visTools.src.sift.sift_aux import *
from visTools.src.sift import sift_descriptor

class SIFT:
    def __init__(self,im='img/house/house0.jpg',Red=2):
        im = plt.imread(im)
        self.im = sf.Reduce(np.mean(im,axis=2),Red)

    def get_xks(self,sig=1.6,N=2,s=5,max_sup_window = (3,3),peak_th=0.03,edge_th=10,ilum_sat=0.2,debug=False,**kwargs):
        im=self.im/255
        im_stack = np.repeat(im[np.newaxis,:,:],s+1,axis=0)
        factor=0
        pyr=[]
        xks=[]
        bin_ = {}
        if debug==True: print('Original image size: {}'.format(im_stack.shape))
        for i in range(N):
            if debug==True: print('factor: {}'.format(i))

            D,L,sigma,L_grad = octave(im_stack,factor=i,sig0=sig,s=s,h=3,w=3)

            xk = np.moveaxis(D[1:-1],0,2)*conv2d_max(D,max_sup_window)

            xk = np.argwhere(xk)
            if debug==True: print('Initial number of kps: {}'.format(xk.shape[0]))
            if debug==True: print('Scale range: {}'.format(np.round(sigma,2)))

            xk = build_sig(xk,sigma)

            deltas,keep = refine_location_prune(im,D,xk,peak_th,edge_th,debug)
            if debug==True: print('Remaining kps after prunning: {}'.format(np.sum(keep)))
            xk[:,:3] += deltas
            xk = xk[keep,:]

            xk = set_rho(L_grad,xk,h=17,w=17)

            bin_[i] = {'xk':xk.copy(),'Lgrad':L_grad}

            xk[:,:2] = xk[:,:2]*(2**i)
            sig*=2

            if debug==True:
                plt.imshow(np.squeeze(im),**{"cmap":"gray"})
                plt.scatter(xk[:,1],xk[:,0],s=1,color='red')
                plt.show()
            xks.append(xk)

        self.xks = np.vstack(xks)
        self.bin_ = bin_

    def get_fks(self,ilum_sat=0.8,block_size=16,cell_size=4,nBins=8,**kwargs):
        fk = []
        for i in self.bin_:
            fk.append(features(self.bin_[i]['Lgrad'],
                          self.bin_[i]['xk'],ilum_sat,
                              block_size,cell_size,nBins))
        self.fks=np.vstack(fk)

    def solve(self,ilum_sat=0.8):
        self.get_xks()
        self.get_fks()
