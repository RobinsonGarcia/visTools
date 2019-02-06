
# coding: utf-8

# In[1]:


import math
import numpy as np
import matplotlib.pyplot as plt
from visTools_v2.core_functions.operations.get_patches import get_patches,get_patches_xks
from visTools_v2.core_functions.corners import Harris_detector
from visTools_v2.core_functions.operations.normalize import norm
from visTools_v2.core_functions.operations.reduce_expand import Reduce_stack
from scipy.special import logsumexp
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# # Helper functions:

# from visTools_v2.core_functions.kernels.kernels import kernel
# from visTools_v2.core_functions.kernels.gaussian_filter import GaussianFilter
# from visTools_v2.core_functions.operations.img_grad import img_grad
# from visTools_v2.core_functions.operations.normalize import norm
# from visTools_v2.core_functions.operations.reduce_expand import Reduce_stack
# from visTools_v2.core_functions.operations.get_patches import *
# 

# In[2]:


def load_img(img,bw=True,Red=None):
    im = plt.imread(img)
    im = np.moveaxis(im,2,0)
    if bw:
        im = np.mean(im,axis=0)
        im = norm(im[np.newaxis,np.newaxis,:,:])
    else:
        im = im[np.newaxis,:,:,:]
    if Red != None:
        im = Reduce_stack(im,Red)
    return im


# ### Recall get_patches...
# get_patches3(im, xk.T,h,w)
# 
# im - N,C,H,W
# 
# xk - n,c,i,j
# 

# In[3]:


def patches(img,h=85,w=85,normalize=True):
    zeros = np.zeros(img.xk.shape[0])
    xk = np.vstack((zeros,zeros,img.xk.T))
    p = get_patches_xks(norm(img.im), xk.T,h,w)
    if normalize:
        num = p - np.mean(p,axis=(1,2,3))[:,np.newaxis,np.newaxis,np.newaxis]
        den = np.std(p,axis=(1,2,3))[:,np.newaxis,np.newaxis,np.newaxis]
        p = np.divide(num,den,out=np.zeros_like(num),where=den!=0)
    return p  


# In[4]:


def plot(img1,img2,id1,id2,s=1):
    plt.figure(figsize=(10,16))
    plt.imshow(img1.im[0,0],**{"cmap":"gray"})
    plt.scatter(img1.xk[:,1][id1],img1.xk[:,0][id1],c="r",s=s)


    for t,i in enumerate(id1):
        plt.text(img1.xk[i,1],img1.xk[i,0],
        str(t),withdash=False,**{'color':'red'})

    plt.figure(figsize=(10,16))
    plt.imshow(img2.im[0,0],**{"cmap":"gray"})
    plt.scatter(img2.xk[:,1][id2],img2.xk[:,0][id2],c="r",s=s)


    for t,i in enumerate(id2):
        plt.text(img2.xk[i,1],img2.xk[i,0],
        str(t),withdash=False,**{'color':'red'})





# In[5]:


def SSD_match(p1_,p2_):
    a_sq = np.sum(p1_**2,axis=(1))
    b_sq = np.sum(p2_**2,axis=(1))
    ab = np.matmul(p1_,p2_.T)
    dist = a_sq[:,np.newaxis]+b_sq - 2*ab

    a_best = np.argmin(dist,axis=1)
    b_best = np.argmin(dist,axis=0)

    idx1 = []
    idx2 = []
    for t,i in enumerate(a_best):
        if b_best[i]==t:
            idx2.append(i)
            idx1.append(b_best[i])


    idx1 = np.array(idx1).astype(np.int)
    idx2 = np.array(idx2).astype(np.int)
    return idx1,idx2



# In[6]:


class SSDMatch:
    def fit(self,x,y):
        a_sq = np.sum(x**2,axis=(1))
        b_sq = np.sum(y**2,axis=(1))
        ab = np.matmul(x,y.T)
        self.dist = a_sq[:,np.newaxis]+b_sq - 2*ab
    def predict(self,X=None,y=None,th=1):        
        a_best = np.argmin(self.dist,axis=1)
        b_best = np.argmin(self.dist,axis=0)

        idx1 = []
        idx2 = []
        d = []
        for t,i in enumerate(a_best):
            if b_best[i]==t:
                idx2.append(i)
                idx1.append(b_best[i])
                d.append(self.dist[b_best[i],i])

        d = norm(np.array(d))

        self.idx1 = np.array(idx1).astype(np.int)[d<th]
        self.idx2 = np.array(idx2).astype(np.int)[d<th]

    def plot(self,img1,img2,s=1):
        plot(img1,img2,self.idx1,self.idx2,s)



# In[7]:


class ExtractFeatures:
    def __init__(self,model = Harris_detector,**params):        
        self.model = model
        self.params = params
        self.w = 31
        self.h = 31
        if isinstance(params,dict):
            for k,v in params.items():
                self.__dict__[k]=v


    def set_params(self,params):
        for k,v in params.items():
            self.__dict__[k]=v

    def fit(self,im1,im2,kmeans_params=None):

        params = self.params

        img1 = self.model(**params)

        img1.fit(im1,kmeans_params)

        img2 = self.model(**params)
        img2.fit(im2,kmeans_params)

        self.p1 = patches(img1,self.h,self.w)
        self.p2 = patches(img2,self.h,self.w)

        self.p1_ = self.p1.reshape((-1,self.h*self.w))
        self.p2_ = self.p2.reshape((-1,self.h*self.w))

        self.img1 = img1
        self.img2 = img2


# im1 =  load_img("img/house/house5.jpg")
# im2 =  load_img("img/house/house9.jpg")
# 
# features = ExtractFeatures()
# features.fit(im1,im2)
# 
# match = SSDMatch()
# match.fit(features.p1_,features.p2_)
# match.predict()
# match.plot(features.img1,features.img2,s=1)

# In[ ]:


class FeatureMatcher:
    def __init__(self,extract_model=Harris_detector, match_model=SSDMatch,**params):
        self.features = ExtractFeatures(model=extract_model,**params)
        self.matcher = match_model

    def fit(self,im1,im2,plot=True,s=1,match_th=1,**kmeans_params):

        self.features.fit(im1,im2,**kmeans_params)

        self.match=self.matcher()
        self.match.fit(x=self.features.p1_,y=self.features.p2_)
        self.match.predict(th=match_th)


        self.xk1= self.features.p1_[self.match.idx1,:]
        self.xk2 =self.features.p2_[self.match.idx2,:]
        if plot: self.match.plot(self.features.img1,self.features.img2,s=s)


if __name__=="__main__":

    # In[ ]:


    params = {"h":85,"w":85,'Red':2,'th':.0001,'blur_window':(3,3),'sig':1}

    im1 =  load_img("img/dino1.jpg")
    im2 =  load_img("img/dino2.jpg")

    matcher = FeatureMatcher(**params)
    matcher.fit(im1,im2,match_th=.4)


    # In[ ]:


    params={"h":85,"w":85,"th":.05}

    im1 =  load_img("img/house/house5.jpg",Red=2)
    im2 =  load_img("img/house/house9.jpg",Red=2)


    matcher = FeatureMatcher(**params)
    matcher.fit(im1,im2,match_th=.5)


    # In[ ]:


    params={"h":85,"w":85,"th":.05}

    im1 =  load_img("img/house/house1.jpg",Red=2)
    im2 =  load_img("img/house/house9.jpg",Red=2)

    matcher = FeatureMatcher(**params)
    matcher.fit(im1,im2,match_th=.5)


    # In[ ]:


    params={"h":51,"w":51,"th":.05}

    im1 =  load_img("img/house/house1.jpg",Red=2)
    im2 =  load_img("img/house/house5.jpg",Red=2)

    matcher = FeatureMatcher(**params)
    matcher.fit(im1,im2,match_th=.5)


    # In[ ]:


    params={"h":55,"w":55,"th":.001,"s":10}

    im1 =  load_img("img/pepsi1.jpg",Red=1)
    im2 =  load_img("img/pepsi2.jpg",Red=1)

    matcher = FeatureMatcher(**params)
    matcher.fit(im1,im2,match_th=.2)


    # In[ ]:


    params={"h":85,"w":85,"th":.01,"s":10,"alpha":.004}

    im1 =  load_img("img/uoft5.jpg",Red=2)
    im2 =  load_img("img/uoft6.jpg",Red=2)

    matcher = FeatureMatcher(**params)
    matcher.fit(im1,im2,match_th=1)


