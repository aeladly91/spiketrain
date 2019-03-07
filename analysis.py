from helper import *

data = load_and_shape('data/spikeTimes_1980_sec.npy')
plot_events(data, 50)
print_gap_stats(data)