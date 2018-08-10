import numpy as np
import matplotlib.pyplot as plt
import math
#https://www.cis.rit.edu/~cnspci/references/dip/feature_extraction/harris1988.pdf

def gradientG(t,sigma):
    return (-t/(2*math.pi*sigma**4))*math.e**(-1*(t**2)/(2*sigma**2))

def DOG(h,w,sigma,grad_type='x'):
    m = (w-1)/2
    n = (h-1)/2
    G = []
    for j in range(w):
        for i in range(h):
            if grad_type=='x':
                G.append(gradientG((i-n),sigma))
            elif grad_type=='y':
                G.append(gradientG((j-n),sigma))

    return np.array(G).reshape(w,h)

def plot(img,**kwargs):
    plt.figure()
    plt.imshow(img,interpolation="none",**kwargs)
    plt.axis('off')

def compute_H(f,Ix,Iy,h=3,w=3,k=1):
    x_2 = Ix**2
    y_2 = Iy**2
    xy = Ix*Iy


    pts = np.argwhere(f)
    H,W = f.shape
    H = int(H)
    W = int(W)

    x_pad = int(0.5*((H-1)*k-H+h))
    y_pad = int(0.5*((W-1)*k-W+w))


    Ix_pad = np.pad(Ix,((y_pad,y_pad),(x_pad,x_pad)),mode='constant')
    Iy_pad = np.pad(Iy,((y_pad,y_pad),(x_pad,x_pad)),mode='constant')

    l1 = []
    l2 = []
    R = []
    for m in range(int(Ix_pad.shape[0]-h)+1):
        for n in range(int(Ix_pad.shape[1]-w+1)):

            x = np.sum(Ix_pad[m:m+w,n:n+h]**2)
            y = np.sum(Iy_pad[m:m+w,n:n+h]**2)
            xy = np.sum(Ix_pad[m:m+w,n:n+h]*Iy_pad[m:m+w,n:n+h])
            H_ = np.array([[x,xy],[xy,y]])
            _,s,_ = np.linalg.svd(H_)

            l1.append(s[-1])
            l2.append(s[0])

    return (np.array(l1).reshape(H,W),np.array(l2).reshape(H,W))


def conv2d_loop(f,gx,gy,k=1,**kwargs):
    w,h = gx.shape
    W,H = f.shape

    gx = np.rot90(gx,2)
    gy = np.rot90(gy,2)

    x_pad = int(0.5*((W-1)*k-W+w))
    y_pad = int(0.5*((H-1)*k-H+w))


    f_pad = np.pad(f,((y_pad,y_pad),(x_pad,x_pad)),mode='constant')

    Ix = []
    Iy = []
    for m in range(int(f_pad.shape[0]-w)+1):
        for n in range(int(f_pad.shape[1]-h+1)):
            fgx = np.multiply(f_pad[m:m+w,n:n+h],gx)
            sum_fgx = np.sum(fgx,axis=(0,1))
            Ix.append(sum_fgx)
            fgy = np.multiply(f_pad[m:m+w,n:n+h],gy)
            sum_fgy = np.sum(fgy,axis=(0,1))
            Iy.append(sum_fgy)

    return np.array(Ix).reshape(W,H),np.array(Iy).reshape(W,H)


class Harris:
    def __init__(self):
        self.Mx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        self.My = np.rot90(self.Mx)

    def solve(self,bw_im,th=0.05,h=3,w=3,gaussian=0,sigma=1):
        self.bw_im = bw_im
        self.Ix,self.Iy = conv2d_loop(bw_im,self.Mx,self.My,k=1)

        l1,l2 = compute_H(bw_im,self.Ix,self.Iy,h,w)

        R = l2*l1-.03*(l2+l1)**2

        #plot(R,**{'cmap':'gray'})
        self.l1 = l1
        self.l2 = l2
        self.R = R

        pts = np.nonzero(self.R[:,:]>th*self.R.max())
        plt.imshow(bw_im,**{'cmap':'gray'})
        plt.scatter(pts[1],pts[0],s=1,**{'color':'red'})
        plt.show()

    def apply_th(self,th):
        bw_im = self.bw_im

        pts = np.nonzero(self.R[:,:]>th*self.R.max())
        plt.imshow(bw_im,**{'cmap':'gray'})
        plt.scatter(pts[1],pts[0],s=1,**{'color':'red'})
        plt.show()
