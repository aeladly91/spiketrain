import numpy as np
import matplotlib.pyplot as plt
import lzma
import scipy

def load_and_shape_varanasi(num_neu):
    datafname = 'data/spikeTimes_medium.npy'
    data = np.load(datafname)

    start = data[0][1] # start timepoint

    N = int(max(data[:, 0]) - min(data[:, 0]) + 1) # num neurons
    data = np.array([data[:, 0], data[:, 1] - start]).T # time start at index 0

    lens = []
    for neuron in range(N):
        lens.append(len(data[np.where(data[:, 0] == neuron)][:, 1]))

    num_fires = min(lens) # number of fires for the neuron with the minimium fires (for cutoff)
    # assert max(lens) - num_fires < 10, 'Min number of neuron fires too low!' # check if significantly different from max neuron so don't cut too much off
    firings = np.zeros((N, num_fires)) # (N, T) np array of firings
    for neuron in range(N):
        firings[neuron] = data[np.where(data[:, 0] == neuron)][:, 1][:num_fires]

    a = np.load('data/spikeTimes_meta/synConnections.npz')
    adj = np.array(scipy.sparse.csr_matrix((a['data'], a['indices'], a['indptr']), shape=a['shape']).todense())

    return firings[:num_neu], adj[:num_neu, :num_neu]

# loads data, adj matrix
def load_and_shape_zaytsev(num_neu, num_firings):
    adjfname = 'data/zaytsev_data/prod-pp-ndt-1000-ai-na-120.00m-1395826618.6076007_J.npy.xz'
    datafname = 'data/zaytsev_data/prod-pp-ndt-1000-ai-na-120.00m-1395826618.6076007.gdf.xz'

    data = np.fromstring(lzma.open(datafname).read(), sep='\t').reshape(-1, 2)
    adj = np.load(lzma.open(adjfname))

    firings = np.zeros((num_neu, num_firings))
    for neuron in range(1, num_neu+1):
        firings[neuron-1] = (data[np.where(data[:, 0] == neuron)][:, 1])[:num_firings]

    return firings[:num_neu], adj[:num_neu, :num_neu]

# pass in number of neurons to plot
def plot_events(firings):
    plt.title('Neuronal Firings')
    plt.eventplot(firings)

def print_gap_stats(firings):
    gaps = np.diff(firings)

    # index i is the variance of the spikes for neuron i
    variances = [np.around(np.var(i), decimals=2) for i in gaps]

    # index i is the average spike gap for neuron i
    avggaps = [np.around(np.mean(i), decimals=2) for i in gaps]

    # neuron with the min and max average gap
    print("Min avg gap:\n", min(avggaps))
    print("Max avg gap:\n", max(avggaps))

    # normalized gaps
    gap_norm = []
    for i in range(len(gaps)):
        # take gaps[i] -= mean of gaps[i]
        gap_norm.append([gaps[i][j]-avggaps[i] for j in range(1,len(gaps[i]))])

    # index i of avg_err_p is abs value of gap_norm[i]/gap_mean[i] for neuron i
    # intuitively, entry i is average percent deviation from pattern of neuron i
    avg_err_p = [np.mean(np.abs(gap_norm[i]))/avggaps[i] for i in range(len(gap_norm))]

    # this is average across neurons of average percent deviation of each neuron
    print("avg % err\n", np.mean(avg_err_p))

    # index i is average number of time steps off neuron i is from it's pattern
    avg_err = [np.mean(np.abs(gap_norm[i])) for i in range(len(gap_norm))]
    print("avg err\n", np.mean(avg_err))

    # avg across neurons of standard deviation of the times for that neuron
    print("avg std of times:\n", np.mean(np.sqrt(variances)))
    # avg across neurons of average deviation time from pattern
    print("std of avg diffs:\n", np.sqrt(np.var(avg_err)))

