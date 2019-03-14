from helper import *
import numpy as np
import scipy.io as sio
import sys

DEFAULT_BIN_SIZE = .5
NUM_NEURONS = 50
# usage: python bin.py <FILENAME> <BIN_SIZE>

fname =  sys.argv[1]
bin_size = float(sys.argv[2]) # bin_size in milliseconds, default is .5
assert bin_size >=  .5 and bin_size <= 100000, 'invalid bin size!'


data = load_and_shape(fname + '.npy')

firings = np.zeros((NUM_NEURONS, 200000), dtype=np.int8)
for i in range(NUM_NEURONS):
	for j in range(len(data[0])):
		firings[i][int(data[i][j])] = 1

bin_width = int(bin_size / DEFAULT_BIN_SIZE)
num_bins = int(firings.shape[1] / bin_width)
bins = np.zeros((NUM_NEURONS, num_bins))
for i in range(NUM_NEURONS):
    print(i)
    for j in range(num_bins):
        bins[i][j] = sum(firings[i][(j-1)*bin_width+1:j*bin_width])

sio.savemat(fname + '.mat', {'data': bins})
