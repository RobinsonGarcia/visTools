import numpy as np
import os
from visTools_v2.core_functions.operations.conv2d import conv2d
from visTools_v2.core_functions.kernels.kernels import kernel
import matplotlib.pyplot as plt



def img_histogram(bw_im,plot=True):
    bw_im = bw_im*100

    MaxVal = bw_im.max()
    MaxRow,MaxCol = bw_im.shape

    H = np.zeros(100)

    for i in range(MaxRow):
        for j in range(MaxCol):
            H[int(bw_im[i,j]-1)]+=1

    if plot==True:
        plt.title('Image Hstogram (black/white)')
        plt.plot(np.arange(100)/100,H)

    return H

norm = lambda x:(x - x.min())/(x.max()-x.min())

def connect(bw_im,ll=0.4,ul=1,plot=False):
    g = kernel().gaussian
    smth_im = conv2d(bw_im,g)[0,:,:,:]
    
    smth_im = np.mean(smth_im,axis=0) 
    if plot==True:_ = img_histogram(norm(smth_im))
    MaxRow,MaxCol = smth_im.shape
    bw_mask = np.zeros((MaxRow,MaxCol))
    bw_mask = (smth_im<ul)&(smth_im>ll)*1
    return recursive_connected_components(bw_mask)

def collect_masks(labelled_mask):
    masks = []
    
    for i in np.unique(labelled_mask):

        masks.append(labelled_mask==i)
    return masks




from tqdm import tnrange
from time import sleep

def recursive_connected_components(B):
    import sys
    sys.setrecursionlimit(15000)

    LB = -B
    label = 0
    LB = find_components(LB,label)
    return LB

def find_components(LB,label):
    MaxRow,MaxCol = LB.shape
    for L in tnrange(MaxRow):
        for P in range(MaxCol):
            if LB[L,P]==-1.0:
                label+=1
                LB = search(LB,label,L,P)
                #sleep(.01)
    return LB

def search(LB,label,L,P):

    LB[L,P] = label
    LB_padded = np.pad(LB,((1,1),(1,1)),'constant')

    Nset = LB_padded[L:L+3,P:P+3]


    for i in range(3):
        for j in range(3):
            if (i==1)&(j==1):
                continue
            if Nset[i,j]==-1:
                if (L-1+i<0)|(P-1+j<0):
                    continue

                else:
                    LB=search(LB,label,L-1+i,P-1+j)
    return LB

