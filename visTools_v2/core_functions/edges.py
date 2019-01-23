


from visTools_v2.core_functions.operations.img_grad import img_grad,load_img
from visTools_v2.core_functions.kernels.kernels import kernel

import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
import math

'''
output from img_grad:
S - Imag                                                      
R - Ix,Iy                                                     
n - Ix,Iy/Imag = cos(teta),sin(teta)                          
rho - arctan2(Iy,Ix)                                          
mask - S >= th                                                
'''

class canny(img_grad):
    def __init__(self,th=0.1,blur=True,h=3,w=3,sig=1,sharp=False,g = kernel().fd,Red=0,plot=False): 
        self.th = th
        self.blur = blur
        self.h = h
        self.w = w
        self.sig = sig
        self.sharp = sharp
        self.g = g
        self.Red= Red

    def fit(self,im,nonMaxSup=True):
        self.im = im
        self.get_grad(th=self.th,
            blur=self.blur,
            h=self.h,
            w=self.w,
            sig=self.sig,
            sharp=self.sharp,g=self.g,
            Red=self.Red)
        self.non_sup_mask = self.mask
        if nonMaxSup: self.nonMaxSup()

    def plot(self):
        #plt.imshow(self.non_sup_mask,**{'cmap':'gray'})
        #plt.show()
        plt.imshow(self.mask,**{'cmap':'gray'})


    def nonMaxSup(self):
        n = self.n
        S = self.S
        S_padded = np.pad(S,((1,1),(1,1)),'constant')
        self.kp = kp = np.argwhere(S)


        n0 = n[:,kp[:,0],kp[:,1]].T

        q0 = kp+n0*1/(2*math.sin(math.pi/8)) 
        q1 = kp-n0*1/(2*math.sin(math.pi/8)) 

        q0 = np.round(q0).astype(int)
        q1 = np.round(q1).astype(int)

        maxs = (S[kp[:,0],kp[:,1]]>S_padded[q0[:,0],q0[:,1]])& (S[kp[:,0],kp[:,1]]>S_padded[q1[:,0],q1[:,1]])

        maxs = kp[maxs]
        new_S = np.zeros_like(S)
        new_S[maxs[:,0],maxs[:,1]]=1
        self.mask = new_S

