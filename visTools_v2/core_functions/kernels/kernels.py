import numpy as np

class kernel:
    def __init__(self,F=1,C=1):
        self.fd = {'x':np.array([[0,0,0],[-1,0,1],[0,0,0]])[np.newaxis,np.newaxis,:,:],'y':np.array([[0,0,0],[-1,0,1],[0,0,0]]).T[np.newaxis,np.newaxis,:,:]}

        self.sobel = {'x':np.array([[-1,0,1],[-2,0,2],[-1,0,1]])[np.newaxis,np.newaxis,:,:],'y':np.array([[-1,0,1],[-2,0,2],[-1,0,1]]).T[np.newaxis,np.newaxis,:,:]}
        self.prewitt = {'x':np.array([[-1,0,1],[-1,0,1],[-1,0,1]])[np.newaxis,np.newaxis,:,:],'y':np.array([[-1,0,1],[-1,0,1],[-1,0,1]]).T[np.newaxis,np.newaxis,:,:]}
 

        self.laplacian = np.array([[1,1,1],[1,8,1],[1,1,1]])[np.newaxis,np.newaxis,:,:]
        self.mean = np.ones((3,3))*1/9
        self.mean = self.mean[np.newaxis,np.newaxis,:,:]
        self.gaussian = np.array([[0.077847,0.123317,0.077847],
    [0.123317,0.195346,0.123317],
    [0.077847,0.123317,0.077847]])[np.newaxis,np.newaxis,:,:]
        self.sharp = np.array([[1,1,1],[1,-9,1],[1,1,1]])[np.newaxis,np.newaxis,:,:]

if __name__=='__main__':
    gg = kernel()
    print('sobel {}',gg.sobel)

