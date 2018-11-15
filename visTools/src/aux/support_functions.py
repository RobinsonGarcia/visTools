import numpy as np
import visTools.src.aux.kernels.GaussianFilter as gf

# In[6]:


def scale_zero2one(M):

    min_ = M.min()
    max_ = M.max()
    M = (M-min_)/(max_-min_)

    return M


# In[5]:


def Expand_stack(im_stack,size):
    N,H,W = im_stack.shape
    _,h,w = size
    factor = int(H/h)
    base=np.zeros((N,h,w))
    i = np.arange(H)*factor
    j = np.arange(W)*factor
    j = np.repeat(j[np.newaxis,:],H,axis=0)
    i = np.repeat(i[:,np.newaxis],W,axis=1)
    print(i.shape,j.shape)
    print(base.shape)
    base[:,i,j] = im_stack
    return base

# In[8]:


def filter_stack(sig0,s,h,w):
    k = 2**(1/s)
    sigma = np.power(k,np.arange(s+1))*sig0
    g = []
    for std in sigma:
        g.append(gf.GaussianFilter(h,w,std))
    return np.stack(g),sigma

# In[25]:

def split_tensor(A):
    x = (np.tile(np.tile(np.arange(4),16) + np.repeat(np.arange(0,14,4),16),4).reshape((4,-1)).T).astype(int)
    y = (np.tile(np.repeat(np.arange(0,14,4),4),4)[:,np.newaxis]+np.arange(4)).astype(int)

    k = A.shape[2]

    z = (np.repeat(np.zeros((64,4)),k).reshape((64,4,-1)) + np.arange(k)).astype(int)
    x = np.repeat(x[:,:,np.newaxis],k,axis=2)
    y= np.repeat(y[:,:,np.newaxis],k,axis=2)


    return np.dstack(np.split(A[x,y,z],16)).reshape((4,4,16,-1))


def split_matrix(A):
    x = (np.tile(np.tile(np.arange(4),16) + np.repeat(np.arange(0,14,4),16),4).reshape((4,-1)).T).astype(int)
    y = (np.tile(np.repeat(np.arange(0,14,4),4),4)[:,np.newaxis]+np.arange(4)).astype(int)
    return np.dstack(np.split(A[x,y],16))



# In[26]:


def Reduce_stack(stack,k):

    N,H0,W0 = stack.shape

    H=H0-H0%(2**k)
    stack = stack[:,:H,:]

    W=W0-W0%(2**k)
    stack = stack[:,:,:W]

    i = np.repeat(np.arange(0,H,2**k)[np.newaxis,:],W//(2**k),axis=0)
    j = np.repeat(np.arange(0,W,2**k)[:,np.newaxis],H//(2**k),axis=1)


    return np.moveaxis(stack[:,i,j],2,1)

def Reduce(im,k):
    for j in range(k):
        H,W = im.shape

        Dx = np.zeros((int((H+H%2)/2),H))
        Dx[np.arange(int((H+H%2)/2)),np.arange(0,H,2)]=1

        Dy = np.zeros((int((W+W%2)/2),W))
        Dy[np.arange(int((W+W%2)/2)),np.arange(0,W,2)]=1
        im = (Dx.dot(im)).dot(Dy.T)
    return im
