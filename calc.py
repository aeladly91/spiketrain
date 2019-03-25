from helper import *
import sklearn.metrics
import scipy.sparse
from sklearn.utils import graph_shortest_path
import sklearn.feature_selection
import time
import matplotlib.pyplot as plt
from directed_information import *
np.set_printoptions(4, threshold=np.nan)

def gen_firings(data, bin_size):
    d = np.floor_divide(data, bin_size)

    N, M = d.shape
    indptr = np.array(range(N+1)) * M
    ones = np.ones(N * M, np.int)
    firings = scipy.sparse.csr_matrix((ones, d.flatten(), indptr))

    return firings.toarray()

def genMI(firings):
    num_neu = len(firings)
    MI = np.zeros((num_neu, num_neu))
    for i in range(num_neu):
        for j in range(num_neu):
            MI[i][j] = sklearn.metrics.mutual_info_score(firings[i], firings[j])

    return MI

def genDI(firings):
    DI = compute_DI_mat(firings)
    return DI

# input: MI matrix
# current prediction method: log binning
def predictMI(MI):
    a = -np.log(MI[:NUM_NEU, :NUM_NEU])
    return a

def predictDI(DI):
    a = -np.log(DI)
    return a

def normalize(mat):
    return mat / np.sum(mat) * mat.shape[0] * mat.shape[1]



