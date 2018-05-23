# This file converts the tss data in numpy format to binary values
# Corresponding to each neuron, there is a binary vector whose t^{th} index is 1 
# that neuron was fired at time t. 

import numpy as np
import pdb

# load data file
filename = 'tss2_np.npy'
tss = np.load(filename)

# find maximum value excluding nan
d = int(np.nanmax(tss))
nr, nc = tss.shape

binary = np.zeros((nr, d), dtype=bool)
# convert into a binary 
for i in range(nr):
    for j in range(nc):
        if not np.isnan(tss[i,j]):
            binary[i,int(tss[i,j])-1] = 1  

# save binary file
np.save(filename+'_binary.npy',binary)