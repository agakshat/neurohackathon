import numpy as np
import pdb
from sklearn.neighbors import NearestNeighbors

def chd(tss1, tss2):
    #pdb.set_trace()
    return np.sum(np.min(np.square(tss1-tss2.transpose()),axis=0))

def main():

    tss  = np.load('tss1_np.npy')

    print(tss.shape)
    tss[np.isnan(tss)]=0
    dist = np.zeros((255,255))
    for i in range(255):
        for j in range(255):
            dist[i,j]=chd(tss[i:i+1], tss[j:j+1])
            # nbrs = NearestNeighbors(n_neighbors=1).fit(tss[i:i+1].transpose())
            # distances, _ = nbrs.kneighbors(tss[j:j+1].transpose())
            # dist[i,j]=np.sum(np.abs(distances))

    pdb.set_trace()

    #dist = np.load('dist.npy')


    #l2_dist= np.linalg.norm()



#if __name__=='__main__()':

main()