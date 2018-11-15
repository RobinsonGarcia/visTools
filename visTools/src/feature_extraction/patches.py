
import numpy as np



def get_patches2(octave,kp,h=9,w=9):

    kp = kp.T

    h_ = int((h-1)/2)+1
    w_ = int((w-1)/2)+1
    octave = np.pad(octave,((0,0),(h_,h_),(w_,w_)),'mean')
    mypks=kp[:3].astype(int).T

    patches=[]
    count=0
    for i in mypks:

        p = octave[i[2],i[0]:i[0]+h,i[1]:i[1]+w]

        patches.append(p)
        count+=1
        if p.shape[0]==0:
            print('patch w/ zero shape on kp {}'.format(i))
    return np.dstack(patches)
