import numpy as np
import pdb
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
def chd(tss1, tss2):
    '''
    Used to compute the chamfer distance between two points (This is used to compute clustering distances)
    '''
    #pdb.set_trace()
    return tf.reduce_mean(tf.reduce_min(tf.square(tss1-tss2),axis=0))

def main():
    '''
    The distances between points are computed using the chamfer distance and a Distance matrix is computed and saved for clustering"
    '''
    tss  = np.load('tss1_np.npy')
    print(tss.shape)
    #tss = tss[:,:n]
    tss[np.isnan(tss)]=0
    dist = np.zeros((255,255))
    sess = tf.Session()
    tss1 = tf.placeholder(tf.float32,[1,None])
    tss2 = tf.placeholder(tf.float32,[None,1])
    dis=chd(tss1, tss2)
    for i in range(255):
        for j in range(255):
            
            dist[i,j] = sess.run(dis, feed_dict = {tss1 : np.expand_dims(np.unique(tss[i:i+1]),0), tss2: np.expand_dims(np.unique(tss[j:j+1]),1)})
            #nbrs = NearestNeighbors(n_neighbors=1).fit(tss[i:i+1].transpose())
            #distances, _ = nbrs.kneighbors(tss[j:j+1].transpose())
            #dist[i,j]=np.sum(np.abs(distances))
        print(i)
    np.save('dist.npy',dist)
    pdb.set_trace()

    dist = (dist + dist.transpose())/2

    import scipy.spatial.distance as ssd

    distA = ssd.squareform(dist)

    cl = h.linkage(distA, method='single', metric='euclidean')

    fig = plt.figure()

    dn = h.dendrogram(cl)

    plt.show()
    #dist = np.load('dist.npy')


    #l2_dist= np.linalg.norm()



if __name__ == "__main__":
    main()
