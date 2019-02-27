import numpy as np
import matplotlib.pyplot as plt
class Homog:
    def cart2homog(self,xk):
        ones = np.ones(xk.shape[1])
        return np.vstack((xk,ones))

class Plotter:
    def __init__(self,im_shape=(0,0),**params):
        self.H,self.W = im_shape
        self.s = .5
        self.c = "r"
        if isinstance(params,dict):
            self.set_par(params)

    def set_par(self,params):
        for k,v in params.items():
            self.__dict__[k] = v

    def line(self,l,reverse=False,**params):
        if isinstance(params,dict):
            self.set_par(params)
        a,b,c = l
        if b==0:
            xrange = (-c/a)*np.ones(self.H)
            y = np.linspace(0,self.H,self.H)
        else:
            n = -a/b
            d = -c/b
            f = lambda x:n*x+d
            xrange = np.arange(self.W)
            if reverse:
                y = f(xrange)
            else:
                y = self.H-f(xrange)

        plt.plot(xrange,y,c=self.c,linewidth=self.s)
        pass

    def point(self,p,reverse=False,**params):
        if isinstance(params,dict):
            self.set_par(params)
        if p[2]==0:
            print("point at infinity")
        else:
            xk = p[:2]/p[2]
        if reverse:
            plt.scatter(xk[0],xk[1],c=self.c,s=self.s)
        else:
            plt.scatter(xk[0],self.H-xk[1],c=self.c,s=self.s)
        pass


    def lines(self,l,reverse=False,**params):
        if isinstance(params,dict):
            self.set_par(params)
        for i in range(l.shape[1]):
            self.line(l[:,i],reverse,**params)

    def points(self,p,reverse=False,**params):
        if isinstance(params,dict):
            self.set_par(params)

        for i in range(p.shape[1]):
            self.point(p[:,i],reverse)
