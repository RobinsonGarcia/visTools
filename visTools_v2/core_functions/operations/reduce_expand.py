import numpy as np


def Reduce_stack(stack,k):

    N,_,H0,W0 = stack.shape

    H=H0-H0%(2**k)
    stack = stack[:,:,:H,:]

    W=W0-W0%(2**k)
    stack = stack[:,:,:,:W]

    i = np.repeat(np.arange(0,H,2**k)[np.newaxis,:],W//(2**k),axis=0)
    j = np.repeat(np.arange(0,W,2**k)[:,np.newaxis],H//(2**k),axis=1)

    return np.moveaxis(stack[:,:,i,j],3,2)




def Reduce(im,k):
    for j in range(k):
        H,W = im.shape

        Dx = np.zeros((int((H+H%2)/2),H))
        Dx[np.arange(int((H+H%2)/2)),np.arange(0,H,2)]=1

        Dy = np.zeros((int((W+W%2)/2),W))
        Dy[np.arange(int((W+W%2)/2)),np.arange(0,W,2)]=1
        im = (Dx.dot(im)).dot(Dy.T)
    return im

if __name__=='__main__':
    im = sys.argv[1]
    k = sys.argv[2]

    Reduce(im,k)
