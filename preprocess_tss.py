# This file converts the tss dataset from string format to numpy array

import pickle
import pdb
import re
import numpy as np

tss_filename = 'tss2'
tss_data = open(tss_filename,'rb').read()

line_splitters = [m.start() for m in re.finditer('\n', tss_data)]
line_splitters.insert(0,-1)


all_data = []
for i in range(len(line_splitters)-1):
    row = tss_data[line_splitters[i]+1:line_splitters[i+1]]
    # convert string to float numbers
    row = [x.strip() for x in row.split(',')]

    for i,x in enumerate(row):
        row[i]=float(x)
    # to round the numbers 
    if not np.isnan(row[i]):
        row[i] = int(round(row[i]))

    # pdb.set_trace()
    all_data.append(row)

all_data = np.array(all_data)

# save as npy file
np.save(tss_filename+'_np.npy',all_data)

