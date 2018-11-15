import numpy as np
from numba import jit

def Affine(pairs,num=4):

    idx = np.arange(pairs[0].shape[0])

    i = np.random.choice(idx,num)

    A = np.zeros((2*num,6))

    A[:num,0] = pairs[0][i,0]
    A[:num,1] = pairs[0][i,1]
    A[:num,4] = np.ones(num)

    A[num:,2] = pairs[0][i,0]
    A[num:,3] = pairs[0][i,1]
    A[num:,5] = np.ones(num)


    y = np.zeros(2*num)
    y[:num] = pairs[1][i,0]
    y[num:] = pairs[1][i,1]


    w = np.linalg.inv(A.T.dot(A)).dot(A.T.dot(y))

    return w,affine_error(pairs,w),pairs[0][i],pairs[1][i]


# In[32]:


def affine_error(pairs,w):

    A = np.reshape(w,(2,3))

    x0 = np.vstack((pairs[0][:,:2].T,np.ones(pairs[0].shape[0])))

    x1_ = A.dot(x0)


    return np.sum((x1_ - pairs[1][:,:2].T)**2,axis=0)


# In[33]:


def affine_project(pairs,best_fit,w):

    x0 = np.vstack((pairs[0][best_fit,:2].T,np.ones(np.sum(best_fit))))

    a,b,e,c,d,f = w

    A = np.array([[a,b,e],[c,d,f]])

    return A.dot(x0)



# # Homography LS

# In[34]:

from scipy import linalg


# In[35]:


@jit
def build_A(num, pairs,i):

    A = np.zeros((2*num,9))

    A[:num,0] = pairs[0][i,0]
    A[:num,1] = pairs[0][i,1]
    A[:num,2] = np.ones(num)

    A[:num,6] = -1*pairs[1][i,0]*pairs[0][i,0]
    A[:num,7] = -1*pairs[1][i,0]*pairs[0][i,1]
    A[:num,8] = -1*pairs[1][i,0]


    A[num:,3] = pairs[0][i,0]
    A[num:,4] = pairs[0][i,1]
    A[num:,5] = np.ones(num)

    A[num:,6] = -1*pairs[1][i,1]*pairs[0][i,0]
    A[num:,7] = -1*pairs[1][i,1]*pairs[0][i,1]
    A[num:,8] = -1*pairs[1][i,1]
    return A
@jit
def Homography(pairs,num=4):
    idx = np.arange(pairs[0].shape[0])

    i = np.random.choice(idx,num)

    A = build_A(num,pairs,i)

    U,S,V = linalg.svd(A)

    h = V[-1]



    e = homography_error(pairs,h)

    return h,e,pairs[0][i],pairs[1][i]


# In[36]:


@jit
def homography_error(pairs,h):

    H = np.reshape(h,(3,3))

    x0 = np.vstack((pairs[0][:,:2].T,np.ones(pairs[0].shape[0])))

    x1_ = H.dot(x0)

    x1_ = np.divide(x1_[:2,:],x1_[2,:],out=np.zeros_like(x1_[:2,:]),
                    where=x1_[2,:]!=0)

    return np.sum((x1_ - pairs[1][:,:2].T)**2,axis=0)
