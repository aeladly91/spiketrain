import numpy as np
import time
from tqdm import tqdm

################################################################################

Nx = 2
D = 2
start_ratio = 0

def ctwupdate(countTree, betaTree, eta, index, xt, alpha):
    Nx = len(eta)
    pw = eta
    pw = pw/np.sum(pw)  # pw(1) pw(2) .. pw(M+1)
    index = int(index)
    pe = (countTree[:,index-1]+0.5)/(np.sum(countTree[:,index-1])+Nx/2)
    temp = betaTree[index-1]
    if temp < 1000:
        eta[:-1] = (alpha*temp * pe[0:Nx-1] + (1-alpha)*pw[0:Nx-1] ) / ( alpha*temp * pe[Nx-1] + (1-alpha)*pw[Nx-1])
    else:
        eta[:-1] = (alpha*pe[0:Nx-1] + (1-alpha)*pw[0:Nx-1]/temp ) / ( alpha*pe[Nx-1] + (1-alpha)*pw[Nx-1]/temp)
    countTree[xt,index-1] = countTree[xt,index-1] + 1
    betaTree[index-1] = betaTree[index-1] * pe[xt]/pw[xt]

    return countTree, betaTree, eta

def ctwalgorithm(x, Nx, D):
    n = len(x)
    countTree = np.zeros( ( Nx, (Nx**(D+1) - 1) // (Nx-1) ))
    betaTree = np.ones( (Nx**(D+1) - 1 ) // (Nx-1) )
    Px_record = np.zeros((Nx,n-D))
    indexweight = Nx**np.arange(D)
    offset = (Nx**D - 1) // (Nx-1) + 1

    for i in range(n-D):
        context = x[i:i+D]
        leafindex = np.dot(context,indexweight)+offset
        xt = x[i+D]
        eta = (countTree[0:Nx,leafindex-1]+0.5)/(countTree[Nx-1,leafindex-1]+0.5)
        eta[-1] = 1
        # update the leaf
        countTree[xt,leafindex-1] = countTree[xt,leafindex-1] + 1
        node = np.floor((leafindex+Nx-2)/Nx)

        while node!=0:
            countTree, betaTree, eta = ctwupdate(countTree, betaTree, eta, node, xt, 1/2)
            node = np.floor((node+Nx-2)/Nx)

        eta_sum = np.sum(eta[:-1])+1

        Px_record[:,i] = eta / eta_sum
    return Px_record

def compute_px(X):
    Px = []
    for i in tqdm(range(len(X))):
        Px.append(ctwalgorithm(X[i], Nx, D))   # 2x8
    return Px

################################################################################

def compute_DI(X, Y, px, py, Nx, D, start_ratio):
    XY=X+Nx*Y
    n_data = len(X)

    pxy = ctwalgorithm(XY, Nx**2, D)    # 4x8
    px_xy = np.zeros((Nx,n_data-D))

    for i_x in range(Nx):
        px_xy[i_x,:] = pxy[i_x,:]
        for j in range(1, Nx):
            px_xy[i_x,:] = px_xy[i_x,:] + pxy[i_x+j*Nx,:]

    temp= np.tile(px_xy, (Nx,1))
    py_x_xy = np.divide(pxy, temp)

    # E4
    temp_DI= np.zeros((1,px.shape[1]))
    for iy in range(Nx):
       for ix in range(Nx):
            tmp1 = pxy[ix+iy*Nx,:]
            tmp2 = np.multiply(py[iy,:], px_xy[ix,:])
            temp_DI = temp_DI + np.multiply( tmp1, np.log2(np.divide(tmp1,tmp2)) )
    DI = np.cumsum(temp_DI[int(np.floor(n_data*start_ratio)):])
    return DI

def compute_DI_mat(X):
    DI = np.zeros((X.shape[0], X.shape[0]))
    Px = compute_px(X)

    for i in tqdm(range(len(X))):
        for j in range(len(X)):
            DI[i][j] = compute_DI(X[i], X[j], Px[i], Px[j], Nx, D, start_ratio)[-1]
    return DI

#####################
## Examples of use ##
# #####################

# A = np.array([1, 0, 1, 0, 1, 0, 1, 1, 0, 0])
# B = np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 0])
# X = np.array([A, B])
# Px = compute_px(X)
# DI = compute_DI(A, B, Px[0], Px[1], Nx, D, start_ratio)
# assert np.isclose(DI, np.array([0., 0.01997409, 0.13639612, 0.27790174, 0.45889724, 0.63982325, 1.15659604, 1.22172031])).all()

# A = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
# B = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
# X = np.array([A, B])
# Px = compute_px(X)
# DI = compute_DI(A, B, Px[0], Px[1], Nx, D, start_ratio)
# assert np.isclose(DI, np.array([0.,         0.01997409, 0.13639612, 0.27790174, 0.45889724, 0.63982325, 0.81449543, 0.97804465])).all()

# A = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
# B = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
# X = np.array([A, B])
# Px = compute_px(X)
# DI = compute_DI(A, B, Px[0], Px[1], Nx, D, start_ratio)
# assert np.isclose(DI, np.array([0.,         0.01997409, 0.13639612, 0.27790174, 0.45889724, 0.63982325, 0.81449543, 0.97804465])).all()







