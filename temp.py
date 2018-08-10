#http://www.sci.utah.edu/~gerig/CS6640-F2012/Materials/Canny-Gerig-Slides-updated.pdf
import numpy.ma as ma

axis = {0:[np.array([1,1]),np.array([-1,-1])],
        1:[np.array([0,1]),np.array([0,-1])],
        2:[np.array([-1,1]),np.array([1,-1])],
        3:[np.array([-1,0]),np.array([1,0])],
        4:[np.array([1,1]),np.array([-1,-1])],
        5:[np.array([0,1]),np.array([0,-1])],
        6:[np.array([-1,1]),np.array([1,-1])],
        7:[np.array([-1,0]),np.array([1,0])]}

axis = {0:[np.array([0,2]),np.array([2,0])],
        1:[np.array([0,1]),np.array([2,1])],
        2:[np.array([0,0]),np.array([2,2])],
        3:[np.array([1,0]),np.array([1,2])],
        4:[np.array([0,2]),np.array([2,0])],
        5:[np.array([0,1]),np.array([2,1])],
        6:[np.array([0,0]),np.array([2,2])],
        7:[np.array([1,0]),np.array([1,2])]}

ref = np.arange(math.pi/4,2.25*math.pi,math.pi/4)

def max_supression(f,g,gx,gy,size=(3,3),k=1,th=0.2):

        w,h = size
        xg, yg = int((w-1)/2),int((h-1)/2)

        W,H = f.shape

        x_pad = int(0.5*((W-1)*k-W+w))
        y_pad = int(0.5*((H-1)*k-H+w))


        f_pad = np.pad(f,((y_pad,y_pad),(x_pad,x_pad)),mode='constant')
        g_pad = np.pad(g,((y_pad,y_pad),(x_pad,x_pad)),mode='constant')
        g_xpad = np.pad(gx,((y_pad,y_pad),(x_pad,x_pad)),mode='constant')
        g_ypad = np.pad(gy,((y_pad,y_pad),(x_pad,x_pad)),mode='constant')

        '''
        mask = np.ma.masked_where(g_pad>th*g_pad.mean(),g_pad)

        g_xpad*=mask
        g_ypad*=mask
        g_pad*=mask
        f_pad *=mask
        '''

        new_im = []
        count1 =0
        count2 =0
        for m in range(0,int(f_pad.shape[0])-w+1,k):
            for n in range(0,int(f_pad.shape[1]-h+1),k):
                G = g_pad[m:m+w,n:n+h]
                Gij = G[xg,yg]

                O = f_pad[m:m+w,n:n+h]
                r = O[xg,yg]
                if r<=0:
                    r+=2*math.pi


                idx = np.argsort((ref-r)**2)[:2]


                a1,b1=axis[idx[0]]
                a2,b2=axis[idx[1]]

                a1 = a1.tolist()
                a2 = a2.tolist()
                b1 = b1.tolist()
                b2 = b2.tolist()


                ux = g_xpad[xg,yg]
                uy = g_ypad[xg,yg]

                Ga = (ux*G[a1[0],a1[1]]+(uy-ux)*G[a2[0],a2[1]])*uy
                Gb = (ux*G[b1[0],b1[1]]+(uy-ux)*G[b2[0],b2[1]])*uy



                if (Gij>Ga)&(Gij>Gb):
                    new_im.append(1.0)


                    count1+=1
                else:
                    new_im.append(0.0)
                    count2+=1


        print('non-max: {}'.format(count1))
        print('max: {}'.format(count2))

        return np.array(new_im).reshape(W,H)

np.seterr(divide='ignore')
from scipy import stats
class CannyDetector():
    def __init__(self,im,sigma=3,size=(3,3)):
        self.im = im
        self.sigma = sigma
        self.w,self.h = size

    def gradientG(self,t,w,sigma):
        return (-t/(2*math.pi*sigma**4))*math.e**(-1*(t**2+w**2)/(2*sigma**2))

    def DOG(self,w,h,sigma,grad_type='x'):
        m = (w-1)/2
        n = (h-1)/2
        G = []
        for i in range(w):
            for j in range(h):
                if grad_type=='y':
                    G.append(gradientG((i-n),(j-n),sigma))
                elif grad_type=='x':
                    G.append(gradientG((j-n),(i-n),sigma))

        return np.array(G).reshape(w,h)

    def conv2d(self,f,g,k=1,**kwargs):
        w,h = g.shape
        W,H,D = f.shape

        g = np.rot90(g,2)

        x_pad = int(0.5*((W-1)*k-W+w))
        y_pad = int(0.5*((H-1)*k-H+w))

        g = np.repeat(g[:,:,np.newaxis],3,2)

        f_pad = np.pad(f,((y_pad,y_pad),(x_pad,x_pad),(0,0)),mode='constant')

        new_im = []
        for m in range(int(f_pad.shape[0]-w)+1):
            for n in range(int(f_pad.shape[1]-h+1)):
                fg = np.multiply(f_pad[m:m+w,n:n+h,:],g)
                sum_fg = np.sum(fg,axis=(0,1))
                new_im.append(sum_fg)

        return np.array(new_im).reshape(W,H,D)

    def edge_strength(self,grad_x,grad_y):
        return np.sqrt(np.power(grad_x,2)+np.power(grad_y,2))

    def gradient_direction(self,grad_x,grad_y):
        return np.arctan(grad_y/grad_x)

    def solve(self,w,h,sigma,th=0):
        gx =  self.DOG(w,h,sigma,'x')
        gy =  self.DOG(w,h,sigma,'y')
        self.grad_x = self.conv2d(self.im,gx)
        self.grad_y = self.conv2d(self.im,gy)
        self.mag = self.edge_strength(self.grad_x,self.grad_y)
        mask = np.ma.masked_where(self.mag>th*self.mag.mean(),self.mag).mask
        self.mag *=mask
        self.orient = self.gradient_direction(self.grad_x*mask,self.grad_y*mask)
        self.mask = max_supression(np.mean(self.orient,axis=2),
                      np.mean(self.mag,axis=2),
                      np.mean(self.grad_x,axis=2),
                      np.mean(self.grad_y,axis=2),
                      size=(3,3),k=1,th=th)


    def plot1(self):
        plot(np.mean(self.mag,axis=2),**{'cmap':'gray'})

    def plot2(self):
        plot(self.mask,**{'cmap':'gray'})




from scipy import interpolate
class interp_matrix():
    def __init__(self,M):
        self.M = M

    def interp(self,M,type_='cubic'):
        w,h = M.shape
        x = np.arange(0,w,1)
        y = np.arange(0,h,1)
        xx, yy = np.meshgrid(x, y)
        z = M
        return interpolate.interp2d(xx, yy, z, kind=type_)

    def get1d(self,ux,uy,xg,yg):
        M = self.M
        if (len(M[M>0.05])<=16)&(len(M[M>0.05])>=4):
            self.f = self.interp(self.M,'linear')
            self.path = np.array([(-2*xg,-2*yg) + np.array((ux,uy))*i for i in np.arange(0,5,.01)])

            self.z = np.array([self.solve(i) for i in self.path])

            return self.z.max()

        elif len(M[M>0.05])<4:
            return 0
        elif len(M[M>0.05])>36:
            self.f = self.interp(self.M,'quintic')
            self.path = np.array([(-5*xg,-5*yg) + np.array((ux,uy))*i for i in np.arange(0,10,.01)])

            self.z = np.array([self.solve(i) for i in self.path])
            return self.z.max()


        else:
            self.f = self.interp(self.M,'cubic')
            self.path = np.array([(-5*xg,-5*yg) + np.array((ux,uy))*i for i in np.arange(0,10,.01)])

            self.z = np.array([self.solve(i) for i in self.path])
            return self.z.max()

    def solve(self,x):
        x,y=x
        return self.f(x,y)



                        g_max = np.max([G[a],G[b],Gij])


                box_gradx = g_xpad[m:m+w,n:n+h]
                box_grady = g_ypad[m:m+w,n:n+h]



                ux = box_gradx[xg,yg]
                uy = box_grady[xg,yg]

                Ga = uy*(ux*G[xg+a,yg+b]+(uy-ux)*G[xg+c,yg+d])
                Gb = uy*(ux*G[xg-a,yg-b]+(uy-ux)*G[xg-c,yg-d])



#http://www.sci.utah.edu/~gerig/CS6640-F2012/Materials/Canny-Gerig-Slides-updated.pdf
import numpy.ma as ma
from scipy.optimize import minimize
def max_supression(f,g,gx,gy,size=(3,3),k=1):
        w,h = size
        xg, yg = int((w-1)/2),int((h-1)/2)

        W,H = f.shape

        x_pad = int(0.5*((W-1)*k-W+w))
        y_pad = int(0.5*((H-1)*k-H+w))


        f_pad = np.pad(f,((y_pad,y_pad),(x_pad,x_pad)),mode='constant')
        g_pad = np.pad(g,((y_pad,y_pad),(x_pad,x_pad)),mode='constant')
        g_xpad = np.pad(gx,((y_pad,y_pad),(x_pad,x_pad)),mode='constant')
        g_ypad = np.pad(gy,((y_pad,y_pad),(x_pad,x_pad)),mode='constant')

        new_im = []
        count1 = 0
        count2 =0
        for m in range(0,int(f_pad.shape[0])-w+1,k):
            for n in range(0,int(f_pad.shape[1]-h+1),k):
                G = f_pad[m:m+w,n:n+h]
                ux = g_xpad[m:m+w,n:n+h][xg,yg]
                uy = g_ypad[m:m+w,n:n+h][xg,yg]

                O = g_pad[m:m+w,n:n+h]


                #mask = ma.masked_where((O[xg,yg]+0.5>O)&
                #               (O[xg,yg]-0.5<O), O).mask

                #G = G*mask


                fun = interp_matrix(G)

                g_max = fun.get1d(ux,uy,xg,yg)

                #f_max = minimize(fun.solve, [0.5,0.5], bounds=((0,None),(0,None)))


                if G[xg,yg]>g_max:
                    new_im.append(1.0)
                    count1+=1
                else:
                    new_im.append(0.0)
                    count2+=1


        print('non-max: {}'.format(count1))
        print('max: {}'.format(count2))

        return np.array(new_im).reshape(W,H)
