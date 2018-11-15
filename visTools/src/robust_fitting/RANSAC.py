
import numpy as np
import matplotlib.pyplot as plt
import visTools.src.CamGeometry.CamGeom as cg




# # Random Sample Consensus - RANSAC
#
# Likelihood that S trials will fail:
# $$1-P=(1-p^k)^s$$
#
# For a given probability of having inliers, the required minimum number of trials S is:
# $$S = \frac{log(1-P)}{log(1-p^k)}$$

# In[38]:


'''
p - probability of being an inlier
k - number of samples
S - number of required trials
'''

def RANSAC(im1,im2,pairs,min_pts=4,p=0.1,P=.99,th=5,type_='affine'):

    S = np.round(np.log(1-P)/(np.log(1-p**min_pts))).astype(int)
    num_pts = pairs[0].shape[0]

    best_fit=0

    for i in range(S):
        try:
            if type_=='affine':
                w,e,x1,x2 = cg.Affine(pairs,min_pts)
            else:
                w,e,x1,x2 = cg.Homography(pairs,min_pts)

            inliers = np.sum(e<th**2)

            if inliers>=best_fit:
                best_w = w
                best_e = np.sum(e)
                best_x1 = x1
                best_x2 = x2
                idx = e<th**2
                best_fit = inliers
                p = inliers/num_pts
                S = np.round(np.log(1-P)/(np.log(1-p**min_pts))).astype(int)

        except:
            continue

    try:

        if type_=='affine':
            best_w,e,x1,x2 = cg.Affine((pairs[0][idx],pairs[1][idx]),len(pairs[0][idx]))
        else:
            best_w,e,x1,x2 = cg.Homography((pairs[0][idx],pairs[1][idx]),len(pairs[0][idx]))

        best_x1 = x1
        best_x2 = x2


        print("Ransac affine/homography best matches:")
        plt.figure(figsize=(10,10*im1.shape[0]/im1.shape[1]))
        plt.subplot(1,2,1)

        plt.imshow(im1,**{'cmap':'gray'})

        plt.scatter(best_x1[:,1],best_x1[:,0],c='red',s=10)

        for i in range(len(pairs[0][idx])):
            plt.text(best_x1[i,1],best_x1[i,0],str(i),withdash=False,**{'color':'red'})


        #plt.show()
        plt.subplot(1,2,2)
        plt.imshow(im2,**{'cmap':'gray'})
        plt.scatter(best_x2[:,1],best_x2[:,0],c='red',s=10)


        for i in range(len(pairs[0][idx])):
            plt.text(best_x2[i,1],best_x2[i,0],str(i),withdash=False,**{'color':'red'})

        plt.savefig('matched.jpeg')
        plt.show()

        print("RANSAC number of inlers: {}".format(best_fit))
    except:
        print("didn't converge")

    if type_=='affine':
        H = np.reshape(best_w,(2,3))
    else:
        H = np.reshape(best_w,(3,3))

    return H,idx
