import pickle
import numpy as np

import matplotlib.pyplot as plt

from visTools_v2.core_functions.operations.get_patches import get_patches

from PIL import Image

def load_img(im1):
    im1 = Image.open(im1)
    im1 = np.array(im1)
    im1 = np.moveaxis(im1,2,0)[np.newaxis,:,:,:]
    return im1


def init(X,k,method='random_selection'):
    if method=='random_selection':
        idx = np.arange(X.shape[0])
        idx = np.random.permutation(idx)
        cs = X[np.random.choice(idx,k),:]
    if method=='random_clustering':
        idx = np.random.randint(0,k,X.shape[0])
        cs = []
        for i in range(k):
            cluster_id = np.argwhere(idx==i)
            cs.append(np.mean(X[cluster_id,:],axis=0))
        cs = np.squeeze(np.array(cs))
    if method=='k++':
        idx = np.arange(X.shape[0])
        cs = []
        cs.append(X[np.random.choice(idx),:])
        for kk in range(k-1):
            d = dists(X,np.array(cs))
            d = np.min(d**2,axis=1)
            id0 = np.random.choice(idx,p=d/np.sum(d))
            cs.append(X[id0,:])
        cs = np.array(cs)

    return cs


def dists(X,cs,im_shape=(None,None),include_coords=True):
    dists = (X[:,:,np.newaxis] - cs.T[np.newaxis,:,:])**2
    dists = np.sqrt(np.sum(dists,axis=1))
    return dists

def include_coords(X,im_shape,norm=False):
    H,W = im_shape
    coords = np.indices((H,W)).reshape((-1,H*W))
    X = np.hstack((coords.T,X))
    X = np.nan_to_num(X)

    if norm==True:
        X[:,:2] = X[:,:2]/np.sqrt(np.sum(X[:,:2]**2,axis=1))[:,np.newaxis]
        X[:,2:] = X[:,2:]/np.sqrt(np.sum(X[:,2:]**2,axis=1))[:,np.newaxis]

    return X

def cluster_center(k,X,clist):
    cs = []
    for i in range(k):
        idx = np.squeeze(np.argwhere(clist==i))
        if idx.ndim==0:
            cs.append(X[idx,:])
        else:
            cs.append(np.mean(X[idx,:],axis=0))
    return np.array(cs)

class Kmeans:
    def __init__(self,maxIters=1000,**kwargs):
        self.maxIters=maxIters
        self.init_method = kwargs["init_method"]
        self.include_coords = kwargs["include_coords"]
        self.k = kwargs["k"]

    def fit(self,X,y=None):
        k = self.k


        diff=1e+30
        cs0 = init(X,k,method=self.init_method)
        count = 0

        while diff>0:
            clist = np.argmin(dists(X,cs0),axis=1)
            cs = cluster_center(k,X,clist)
            diff = np.sum((cs - cs0)**2)
            cs0 = cs
            count+=1
            if count>self.maxIters:
                print('--reached maximum iters')
                break

        self.clist = clist
        self.cs = cs

    def plot(self):
        colors = ["r","g","b","y","w"]
        ids = np.unique(self.clist)
        for i in ids:
            w,h = np.argwhere(self.clist==i).T
            plt.scatter(h,w,s=.2)#,c=colors[i])
            plt.axis("off")
            plt.imshow(self.clist.astype(np.uint8))

    def predict(self,X=None,y=None):
        return None


