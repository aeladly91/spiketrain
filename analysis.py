from helper import *
import numpy as np
import scipy.io as sio

fname = 'data/spikeTimes_medium'

data = load_and_shape(fname + '.npy')

firings = np.zeros((1000, 200000), dtype=np.int8)
for i in range(len(data)):
	for j in range(len(data[0])):
		firings[i][int(data[i][j])] = 1

sio.savemat(fname + '.mat', {'data': firings})

# plot_events(data, 50)
# print_gap_stats(data)
