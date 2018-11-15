
import numpy as np
import visTools.src.sift.sift as sf
import visTools.src.robust_fitting.RANSAC as ransac
import visTools.src.feature_matching.nn as nn
import matplotlib.pyplot as split_tensor
%matplotlib inline

class Match:
    def __init__(self,im1,im2,kwargs,Red=2):
        self.sift1 = sf.SIFT(im1,Red)
        self.sift2 = sf.SIFT(im2,Red)
        self.sift1.get_xks(**kwargs['sift'])
        self.sift2.get_xks(**kwargs['sift'])
        self.sift1.get_fks(**kwargs['fk'])
        self.sift2.get_fks(**kwargs['fk'])
        self.nnRANSAC(**kwargs['RANSAC'])


    def nn(self,crt=0.8):
        self.pairs,self.idx = nn.nearest_neighboor(self.sift1.fks,
                                                self.sift1.xks,
                                                self.sift2.fks,
                                                self.sift2.xks,
                                                crt)

    def nnRANSAC(self,type_='homography',min_pts=4,P=.98,
                 RANSAC_th=5,RANSAC_p=0.2,**kwargs):
        self.nn()
        H,idx = ransac.RANSAC(self.sift1.im,self.sift2.im,self.pairs,min_pts,type_=type_,P=.98,th=RANSAC_th,p=RANSAC_p)
        self.pairs = (self.pairs[0][idx],self.pairs[1][idx])


import time
start = time.time()

kwargs = {
    'sift':{'N':2,'s':5,'max_sup_window':(5,5)},
    'fk':{'block_size':16},
    'RANSAC':{'RANSAC_th':5,'RANSAC_p':0.3}
        }
MM = Match(im1='img/house/house8.jpg',
           im2='img/house/house23.jpg',
           kwargs=kwargs)

end = time.time()
print('Total time: {}'.format(end-start))
