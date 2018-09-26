import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#https://www.cis.rit.edu/~cnspci/references/dip/feature_extraction/harris1988.pdf

def grad(mode):
    if mode=='fd':
        gx = np.array([[0,0,0],[-1,0,1],[0,0,0]])
        gy = gx.T
        g = np.dstack((gx,gy))
    if mode=='sobel':
        gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        gy = gx.T
        g = np.dstack((gx,gy))
    if mode=='laplacian':
        gx = np.array([[1,1,1],[1,8,1],[1,1,1]])
        gy = gx.T
        g = np.dstack((gx,gy))
    return g

def get_im2col_indices(x_shape, field_height, field_width, p_x=1,p_y=1, stride=1):
    # First figure out what the size of the output should be
    _, C, H, W = x_shape
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



def conv2d(im,g,stride=1,C=3):
    im = np.repeat(im[np.newaxis,:,:],2,axis=0)

    g = np.moveaxis(g,2,0)

    im = im[:,np.newaxis,:,:]
    N,_,H,W = im.shape
    _,h,w = g.shape

    stride=1
    h_pad = int((H*(stride-1)-stride+h)/2)
    w_pad = int((W*(stride-1)-stride+w)/2)

    k,i,j = get_im2col_indices((N,1,H,W), h, w, p_x=h_pad,p_y=w_pad, stride=1)

    im_padded = np.pad(im,((0,0),(0,0),(h_pad,h_pad),(w_pad,w_pad)),'mean')
    cols = im_padded[:,k,i,j]


    g = g.reshape((N,-1))


    sol = np.squeeze(np.matmul(g[:,np.newaxis,:],cols))

    return sol.reshape(N,H,W)

def GaussianFilter(w,h,sigma):
    m = (w-1)/2
    n = (h-1)/2
    G = []
    for i in range(w):
        for j in range(h):
            G.append((1/(2*math.pi*sigma**2))*math.e**(-1*((i-m)**2+(j-n)**2)/(2*sigma**2)))

    return np.array(G).reshape(w,h)/np.sum(np.array(G))

def Reduce(im,k):
    for j in range(k):
        H,W = im.shape

        Dx = np.zeros((int((H+H%2)/2),H))
        Dx[np.arange(int((H+H%2)/2)),np.arange(0,H,2)]=1

        Dy = np.zeros((int((W+W%2)/2),W))
        Dy[np.arange(int((W+W%2)/2)),np.arange(0,W,2)]=1
        im = (Dx.dot(im)).dot(Dy.T)
    return im


class Harris_detector:
    def __init__(self,img):
        self.img = img


    def solve(self,solve=False,blur_window=(3,3),deriv_type='fd',
                                          sig=1,Red=2,sharp=False,
                                          blur=True,
                         second_moment_window=(3,3),
                        clusters=None,
                        th=0.03):

        '''gaussian_derivatives param:'''
        self.blur_window = blur_window
        self.deriv_type= deriv_type
        self.sig = sig
        self.Red= Red
        self.sharp = sharp
        self.blur = blur
        self.second_moment_window=second_moment_window

        '''thresholding parameters'''
        self.th = th
        self.clusters = clusters


        self.compute()


    def compute(self):

        I = self.get_gaussian_derivatives(self.img)
        H = self.get_H(I)
        self.R = self.get_corner_response(H)
        self.apply_threshold()


    def get_gaussian_derivatives(self,img):
        Red=self.Red
        sharp=self.sharp
        blur=self.blur
        sig=self.sig
        h,w=self.blur_window
        deriv_type=self.deriv_type

        self.im = plt.imread(img)
        self.im = Reduce(np.mean(self.im,axis=2),Red)
        self.im_shape = self.im.shape

        if sharp==True:
            g = np.array([[1,1,1],[1,-9,1],[1,1,1]])
            g = np.dstack((g,g.T))
            im = conv2d(self.im,g)[0]

        if blur==True:
            G = GaussianFilter(h,w,sig)
            G = np.dstack((G,G))
            im = conv2d(self.im,G)[0]
        self.im = im
        g = grad(deriv_type)
        R = conv2d(im,g)
        self.Sx = R[0]
        self.Sy = R[1]

        rho = np.arctan2(self.Sy,self.Sx)
        rho[rho<0]+=2*math.pi
        self.rho = rho

        self.S = np.sqrt(np.sum(R**2,axis=0))
        self.n = np.divide(R,self.S[np.newaxis,:,:],out=np.zeros_like(R),where=self.S!=0)
        return R

    '''Second moment matrix'''
    def get_H(self,R):
        Ix,Iy = R
        Ixy = Ix*Iy
        H_ = np.stack((Ix**2,Ixy,Ixy,Iy**2))

        N,H,W = H_.shape
        x_shape = (0,N,H,W) #image shape
        field_height,field_width = self.second_moment_window #Gaussian window

        stride=1
        h_pad = int((H*(stride-1)-stride+field_height)/2)
        w_pad = int((W*(stride-1)-stride+field_width)/2)

        k,i,j = get_im2col_indices(x_shape, field_height, field_width, p_x=h_pad,p_y=w_pad, stride=1)

        H_padded = np.pad(H_,((0,0),(h_pad,h_pad),(w_pad,w_pad)),'mean')

        cols = H_padded[k,i,j]

        H_ = np.sum(cols.reshape(4,field_height*field_width,-1),axis=1)
        H_ = H_.reshape(2,2,-1)

        H_ = np.moveaxis(H_,2,0)
        return H_

    def get_corner_response(self,H_):
        U,S,V = np.linalg.svd(H_)
        alpha=0.04
        R = S[:,0]*S[:,1] - alpha*(S[:,0]+S[:,1])**2
        return R

    def apply_threshold(self):
        th = self.th
        cluster = self.clusters
        H,W = self.im_shape
        R_ = np.clip(self.R,0,None)
        R_ = (R_ - R_.min())/(R_.max()-R_.min())


        R_ = R_.reshape(H,W)

        R_[R_<th]=0

        R_ = self.nonMaxSup(R_)

        xk = np.argwhere(R_)

        if cluster !=None:
            kmeans = KMeans(n_clusters=cluster, random_state=0).fit(xk)

            xk = kmeans.cluster_centers_

        plt.scatter(xk[:,1],xk[:,0],c='r',s=1)
        plt.imshow(self.im,**{'cmap':'gray'})
        for j in range(xk.shape[0]):
            plt.text(xk[j,1],xk[j,0],str(j),withdash=False,**{'color':'red','size':7})

        plt.show()
        self.xk = xk

    def nonMaxSup(self,S):
        n = self.n
        S_padded = np.pad(S,((1,1),(1,1)),'mean')
        kp = np.argwhere(S)

        n0 = n[:,kp[:,0],kp[:,1]].T

        q0 = kp+n0*1/(2*math.sin(math.pi/8))
        q1 = kp-n0*1/(2*math.sin(math.pi/8))

        q0 = np.round(q0).astype(int)
        q1 = np.round(q1).astype(int)

        maxs = (S[kp[:,0],kp[:,1]]>S_padded[q0[:,0],q0[:,1]])| (S[kp[:,0],kp[:,1]]>S_padded[q1[:,0],q1[:,1]])

        maxs = kp[maxs]
        new_S = np.zeros_like(S)
        new_S[maxs[:,0],maxs[:,1]]=1
        return new_S
