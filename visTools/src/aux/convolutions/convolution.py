
import numpy as np
import visTools.src.aux.convolutions.cython_convs as cc

# In[7]:


def conv_octave_cython(im,g,stride=1,C=3):
    im = im[:,np.newaxis,:,:]

    N,_,H,W = im.shape
    _,h,w = g.shape

    h_pad = int((H*(stride-1)-stride+h)/2)
    w_pad = int((W*(stride-1)-stride+w)/2)

    ii = np.zeros((N,H*W))
    for t,i in enumerate(im):
        img = i[np.newaxis,:,:]
        cols = cc.im2col_cython(img, h, w, h_pad,stride)
        gg = g[t].flatten()
        ii[t]=np.matmul(cols.T,gg)
    return ii.reshape(-1,H,W)
