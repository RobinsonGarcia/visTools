import numpy as np
import math

def gradientG(t,w,sigma):
    return (-t/(2*math.pi*sigma**4))*math.e**(-1*(t**2+w**2)/(2*sigma**2))

def DOG(h,w,sigma,grad_type='x'):
    m = (w-1)/2
    n = (h-1)/2
    G = []
    for j in range(w):
        for i in range(h):
            if grad_type=='x':
                G.append(gradientG((i-n),(j-n),sigma))
            elif grad_type=='y':
                G.append(gradientG((j-n),(i-n),sigma))

    return np.array(G).reshape(w,h)

def conv2d_loop(f,gx,gy,k=1,**kwargs):
    w,h = gx.shape
    W,H = f.shape

    gx = np.rot90(gx,2)
    gy = np.rot90(gy,2)

    x_pad = int(0.5*((W-1)*k-W+w))
    y_pad = int(0.5*((H-1)*k-H+w))

    #g = np.repeat(g[:,:,np.newaxis],3,2)

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

def grad(f,h,w,sigma):
    gx = DOG(h,w,sigma,'x')
    gy = DOG(h,w,sigma,'y')

    Mx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    My = np.rot90(Mx)
    return conv2d_loop(f,gx,gy,k=1)

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
