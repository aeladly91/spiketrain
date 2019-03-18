from helper import *
import sklearn.metrics
import scipy.sparse
from sklearn.utils import graph_shortest_path
import sklearn.feature_selection
import time
import matplotlib.pyplot as plt
np.set_printoptions(4, threshold=np.nan)

NUM_NEU = 10 # num neurons to test
BIN_SIZES = [500] # bin sizes to test

# Creates a MI (mutual information matrix)
# input: bin size -- the number of indices per bin
def genMI(data, bin_size):
    d = np.floor_divide(data, bin_size)

    N, M = d.shape
    indptr = np.array(range(N+1)) * M
    ones = np.ones(N * M, np.int)
    firings = scipy.sparse.csr_matrix((ones, d.flatten(), indptr))
    MI = np.zeros((NUM_NEU, NUM_NEU))
    for i in range(NUM_NEU):
        for j in range(NUM_NEU):
            MI[i][j] = sklearn.metrics.mutual_info_score(np.array(firings[i].todense()).flatten(), np.array(firings[j].todense()).flatten())

    return MI


# input: MI matrix
# current prediction method: log binning
def predict(MI):
    a = -np.log(MI[:NUM_NEU, :NUM_NEU])
    _, bins = np.histogram(a.flatten(), bins=[0, 2, 4, 7.5, 10, 12]) # hardcoded bins
    return np.digitize(a, bins) - 1

# load and convert dist matrix
a = np.load('data/spikeTimes_meta/synConnections.npz')
adjmat = scipy.sparse.csr_matrix((a['data'], a['indices'], a['indptr']), shape=a['shape'])
distmat = graph_shortest_path.graph_shortest_path(adjmat, directed=True)

# load and convert data matrix
data = load_and_shape('data/spikeTimes_medium.npy')[:NUM_NEU] # right now this shaves off end firings, but we can fix this

# calculate MIs
for i in range(len(BIN_SIZES)):
    MI = genMI(data, BIN_SIZES[i])
    pred = predict(MI)
    print(pred)
    truth = distmat[:NUM_NEU, :NUM_NEU]
    print(truth)

    # error printing
    num_bins = int(max(data[:, -1]) / BIN_SIZES[i])
    error = np.count_nonzero(pred - truth) / (pred.shape[0] * pred.shape[1])
    print("num bins: " + str(num_bins) + ', percent error: ' + str(error))

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(pred, cmap='hot', interpolation='nearest')
    plt.subplot(1, 2, 2)
    plt.imshow(truth, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()


