
# ## Pyramid of DOG images:
# $$D(x,y,\rho) = I(x,y)*(G(x,y,k\rho) - G(x,y,\rho))$$
# for $$\rho=\sigma.k\sigma,k^2\sigma,...,k^{s-1}\sigma$$
import numpy as np
import visTools.src.aux.support_functions as sf
import visTools.src.aux.convolutions.convolution as cv
# In[9]:


def scale_zero2one(M):

    min_ = M.min()
    max_ = M.max()
    M = (M-min_)/(max_-min_)

    return M

# In[10]:


def octave(im_stack,factor,sig0=1.6,s=5,h=3,w=3):
    g,sigma = sf.filter_stack(sig0,s,h,w)
    s1 = np.array([[0,1,0],[0,0,0],[0,-1,0]])
    s2 = np.repeat(s1.T[np.newaxis,:,:],s+1,axis=0)
    s1 = np.repeat(s1[np.newaxis,:,:],s+1,axis=0)

    L = cv.conv_octave_cython(sf.Reduce_stack(im_stack,factor),g)
    L = sf.scale_zero2one(L)

    D = L[1:s+1] - L[0:s]

    L_gradx = cv.conv_octave_cython(L,s2)
    L_grady = cv.conv_octave_cython(L,s1)

    return D,L,sigma,(L_gradx,L_grady)
