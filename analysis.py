from helper import *
import scipy.io as sio

data = load_and_shape('data/spikeTimes.npy')
sio.savemat('data/spikeTimes.mat', {'data': data})

# plot_events(data, 50)
# print_gap_stats(data)