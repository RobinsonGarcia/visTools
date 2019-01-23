import numpy as np
import math
import matplotlib.pyplot as plt

from visTools_v2.core_functions.operations.normalize import norm
from visTools_v2.core_functions.kernels.kernels import kernel
from visTools_v2.core_functions.kernels.gaussian_filter import GaussianFilter
from visTools_v2.core_functions.operations.conv2d import conv2d
from visTools_v2.core_functions.operations.reduce_expand import Reduce,Reduce_stack

def load_img(img):
    im = plt.imread(img)
    plt.imshow(im)
    plt.show()
    im = np.moveaxis(im,2,0)
    im = np.mean(im,axis=0)
    im = im[np.newaxis,np.newaxis,:,:]
    return im
'''
S - Imag
R - Ix,Iy
n - Ix,Iy/Imag = cos(teta),sin(teta)
rho - arctan2(Iy,Ix)
mask - S >= th
'''

class img_grad:
    def __init__(self,im,plot=False):
        self.im = im
        self.get_grad()

    def get_grad(self,sharp=False,blur=True,h=5,w=5,sig=1,th=0.03,Red=0,g=kernel().sobel):

        im = norm(Reduce_stack(self.im,Red))

        if sharp==True:
            im = conv2d(im,kernel().sharp)

        if blur==True:
            G = GaussianFilter(h,w,sig)[np.newaxis,np.newaxis,:,:]
            im = conv2d(im,G)

        R = np.array([conv2d(im,g['x'])[0,0,:,:],conv2d(im,g['y'])[0,0,:,:]])
        S = np.sqrt(np.sum(R**2,axis=0))

        mask = norm(S) 
        mask[mask>=th] = 1
        mask[mask<th] = 0

        rho = np.arctan2(R[1],R[0])
        rho[rho<0]+= 2*math.pi

        n=np.divide(R,S[np.newaxis,:,:],out=np.zeros_like(R),where=S!=0)
        n*=mask
        self.n = n
        self.S = S*mask
        self.mask = mask
        self.R = R[0]*mask,R[1]*mask

    def vector_field(self,figsize=(10,16),tan=False):
        mask = self.mask
        n = self.n
        h,w = mask.shape
        x,y = np.nonzero(mask)
        idx = np.arange(x.shape[0])
        idx = np.random.choice(idx,10)
        x = x[idx]
        y = y[idx]
        nx,ny = n[:,x,y]
        plt.figure(figsize=figsize)
        plt.imshow(mask,**{'cmap':'gray'})#,extent=[0,w,0,h])
        plt.quiver(y,x,ny,nx,color='r',scale_units='xy', angles='xy', scale=None,width=0.01)
        if tan ==True:
            n = np.array([-self.n[1],self.n[0]])
            nx,ny = n[:,x,y]
            plt.quiver(y,x,ny,nx,color='g',scale_units='xy', angles='xy', scale=None,width=0.01)

    def get_tan(self):
        return np.array([-self.n[1],self.n[0]])

