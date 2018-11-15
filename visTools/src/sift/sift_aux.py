
import numpy as np
import visTools.src.feature_extraction.patches_cython as gp
from visTools.src.aux.convolutions import convolution as cv
import visTools.src.sift.sift_descriptor as sd
import visTools.src.aux.histogram as histogram
import math

# In[13]:
def features(L_grad,xk,ilum_sat=0.2
             ,block_size=16,cell_size=4,nBins=8,**kwargs):
    N = xk.shape[0]
    Lx,Ly = L_grad
    Lmod = np.sqrt(Lx**2+Ly**2)
    rho = np.arctan2(Ly,Lx)
    rho[rho<0]+=2*math.pi

    fks = sd.build_fk(Lmod,rho,xk,block_size,cell_size,nBins)

    mod = np.sqrt(np.sum(fks**2,axis=1))[:,np.newaxis]
    fks = np.divide(fks,mod,out=np.zeros_like(fks),where=mod!=0)
    fks = np.clip(fks,a_min=None,a_max=ilum_sat)
    mod = np.sqrt(np.sum(fks**2,axis=1))[:,np.newaxis]
    fks = np.divide(fks,mod,out=np.zeros_like(fks),where=mod!=0)

    return fks

def set_rho(Lgrad,xk,h,w):

    Lgradx,Lgrady = Lgrad
    Lmod =np.sqrt(Lgradx**2+Lgrady**2)
    Lrho = np.arctan2(Lgrady,Lgradx)
    Lrho[Lrho<0]+=2*math.pi
    Lmod = gp.get_patches3(Lmod,xk,h,w)
    Lrho = gp.get_patches3(Lrho,xk,h,w)
    bins = np.arange(0,2*math.pi+math.pi/36,2*math.pi/36)
    hist = histogram.histogram(Lmod.reshape((Lmod.shape[0],-1)),
                                   Lrho.reshape((Lrho.shape[0],-1)),
                                   bins,xk,8)

    mod = np.sum(hist**2,axis=1)[:,np.newaxis]
    hist=np.divide(hist,mod,out=np.zeros_like(hist),where=mod!=0)
    idx = np.argmax(hist,axis=1)
    rho = bins[idx]

    return np.vstack((xk.T,rho)).T

# ## Detecting DOG extrema
#
# Find all pixels that correspond to extrema of $D(x,y,\rho)$. For each $(x,y,\rho)$, check wether $D(x,y,\rho)$ is greater than (or smaller than) all of its neighbours in current scale and adjacent scales above & below.

# In[15]:


def get_im2col_indices_conv2d_max(x_shape, field_height, field_width, p_x=1,p_y=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * p_x - field_height) % stride == 0
    assert (W + 2 * p_y - field_height) % stride == 0
    out_height = int((H + 2 * p_x - field_height) / stride + 1)
    out_width = int((W + 2 * p_y - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def conv2d_max(f,g,k=1,**kwargs):
    #f = f[:,np.newaxis,:,:]
    #f = np.repeat(f,3,axis=1)

    N,H,W=f.shape

    n = np.tile(np.arange(3),N-2) + np.repeat(np.arange(N-2),3)
    f = f[n,:,:].reshape((N-2,3,H,W))

    N,C,H,W = f.shape
    h,w = g

    x_pad = int(0.5*((W-1)*1-W+w))
    y_pad = int(0.5*((H-1)*1-H+w))

    f_pad = np.pad(f,((0,0),(0,0),(x_pad,x_pad),(y_pad,y_pad)),mode='maximum')

    k,i,j = get_im2col_indices_conv2d_max((N,C,H,W), h, w, p_x=x_pad,p_y=y_pad, stride=1)

    cols = f_pad[:,k,i,j]

    xc = int((h*w-1)/2)
    cols = np.delete(cols,(xc,xc+h*w,xc+2*h*w),1)

    cond = ((f[:,1,:,:].flatten() >
             np.max(cols,axis=1).flatten())|(f[:,1,:,:].flatten() <
                                   np.min(cols,axis=1).flatten()))

    return np.moveaxis(cond.reshape(N,H,W),0,2)

# In[16]:


def build_sig(xk,sigma):
    sig = np.zeros(xk.shape[0])
    k=0
    for i in xk:
        sig[k] = sigma[i[2]]
        k+=1
    return np.vstack((xk.T,sig)).T



# In[17]:


def HOG(L_grad,xk,h=9,w=9):
    Lx,Ly = L_grad
    Lmod = np.sqrt(Lx**2+Ly**2)
    rho = np.arctan2(Ly,Lx)
    rho[rho<0]+=2*math.pi
    Lmod_patches = gp.get_patches3(Lmod,xk,h,w)
    rho_patches = gp.get_patches3(rho,xk,h,w)
    return Lmod_patches,rho_patches


# In[18]:


def harris2(D,xk,edge_th):

    #gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    gy = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    gx = gy.T
    gx = np.repeat(gx[np.newaxis,:,:],D.shape[0],axis=0)
    gy = np.repeat(gy[np.newaxis,:,:],D.shape[0],axis=0)

    D2grad_x = cv.conv_octave_cython(D,gx)
    D2grad_y = cv.conv_octave_cython(D,gy)

    Dxx = np.sum(gp.get_patches3(D2grad_x,xk),axis=(1,2))
    Dyy = np.sum(gp.get_patches3(D2grad_y,xk),axis=(1,2))

    H = np.array([[Dxx**2,Dxx*Dyy],[Dxx*Dyy,Dyy**2]])

    tr = np.trace(H)**2
    det = np.linalg.det(np.moveaxis(H,2,0))

    #R = det - 0.04*tr
    return np.divide(tr,det,out=np.zeros_like(tr),where=det!=0)<((edge_th+1)/10)**2




# In[19]:

def get_deriv(pyr,kp):
    pyr = np.moveaxis(pyr,0,2)

    z = kp[:,2].astype(int)
    y = kp[:,1].astype(int)+1
    x = kp[:,0].astype(int)+1


    pyr = np.pad(pyr,((1,1),(1,1),(0,0)),'maximum')

    Ix = pyr[x+1,y,z] - pyr[x-1,y,z]
    Iy = pyr[x,y+1,z] - pyr[x,y-1,z]
    Is = pyr[x,y,z+1] - pyr[x,y,z-1]
    Ixy = pyr[x+1,y+1,z] - pyr[x-1,y-1,z]

    #H = np.array([[Ix,Ixy],[Ixy,Iy]])

    Ixx = pyr[x+1,y,z] - 2*pyr[x,y,z] + pyr[x-1,y,z]
    Iyy = pyr[x,y+1,z] - 2*pyr[x,y,z] + pyr[x,y-1,z]
    Iss = pyr[x,y,z+1] - 2*pyr[x,y,z] + pyr[x,y,z-1]
    Ixxyy = pyr[x+1,y+1,z] - 2*pyr[x,y,z] + pyr[x-1,y-1,z]
    Ixxss = pyr[x+1,y,z+1] - 2*pyr[x,y,z] + pyr[x-1,y,z-1]
    Iyyss = pyr[x,y+1,z+1] - 2*pyr[x,y,z] + pyr[x,y-1,z-1]

    H = np.array([[Ixx,Ixxyy,Ixxss],[Ixxyy,Iyy,Iyyss],[Ixxss,Iyyss,Iss]])
    J = np.array([[Ix**2,Ix*Iy],[Ix*Iy,Iy**2]])
    dx = np.array([Ix,Iy,Is])
    return H,dx,J




# In[20]:



def refine_location_prune(im,D,kp,peak_th=0.03,edge_th=10,debug=False):
    D = D**2
    #D = (D-D.min())/(D.max()-D.min())

    H,dx,J = get_deriv(D,kp)

    U,S,V = np.linalg.svd(np.moveaxis(H,2,0))
    S = S[:,:,np.newaxis]*np.diag(np.ones(3))
    A = np.linalg.inv(S)
    H_inv = np.matmul(-np.matmul(A,S),V)
    deltas = np.squeeze(np.matmul(H_inv,np.moveaxis(dx,1,0)[:,:,np.newaxis]))
    #deltas=np.round(deltas)

    xk=kp
    D[xk[:,2].astype(int),xk[:,0].astype(int),xk[:,1].astype(int)]+=0.5*np.sum(dx.T*deltas,axis=1)

    Ds = D[xk[:,2].astype(int),xk[:,0].astype(int),xk[:,1].astype(int)]
    Ds = (Ds-Ds.min())/(Ds.max()-Ds.min())
    if debug==True: print('Maximum contrast: {}'.format(Ds.max()))
    if debug==True: print('Minimum contrast: {}'.format(Ds.min()))
    if debug==True: print('Average contrast: {}'.format(Ds.mean()))
    cond1 = Ds > peak_th
    if debug==True: print("Number of kps removed after th: {}".format(np.sum(~cond1)))
    #if kp.shape[1]!=0:
    cond2 = harris2(D,kp,edge_th)
    if debug==True: print("Number of kps removed edge extraction: {}".format(np.sum(~cond2)))

    keep = ((cond2) & (cond1))
    return (deltas,keep)
