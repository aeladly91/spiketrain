a = np.load('data/spikeTimes_meta/synConnections.npz')

data = a['data']

indices = a['indices']

indptr = a['indptr']

shape = a['shape']

matrix = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape).todense()

matrix = np.squeeze(np.asarray(M))
matrix = np.resize(matrix,(50,50))