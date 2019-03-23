import numpy as np

################################################################################

def ctwupdate(countTree, betaTree, eta, index, xt, alpha):
    le = len(eta)
    Nx = le+1
    pw = eta
    pw = np.append(pw, [1])
    pw = pw/np.sum(pw)  # pw(1) pw(2) .. pw(M+1)
    index = int(index)
    pe = (countTree[:,index-1]+0.5).T/(np.sum(countTree[:,index-1])+Nx/2)
    temp = (betaTree[index-1])
    if temp < 1000:
        eta = (alpha*temp * pe[0:Nx-1] + (1-alpha)*pw[0:Nx-1] ) / ( alpha*temp * pe[Nx-1] + (1-alpha)*pw[Nx-1])
    else:
        eta = (alpha*pe[0:Nx-2] + (1-alpha)*pw[0:Nx-2]/temp ) / ( alpha*pe[Nx-1] + (1-alpha)*pw[Nx-1]/temp)
    countTree[xt,index-1] = countTree[xt,index-1] + 1
    betaTree[index-1] = betaTree[index-1] * pe[xt]/pw[xt]
    return countTree, betaTree, eta

def ctwalgorithm(x, Nx, D):
    n=len(x)
    countTree = np.zeros( ( Nx, (Nx**(D+1) - 1) // (Nx-1) ))
    betaTree = np.ones( (Nx**(D+1) - 1 ) // (Nx-1) )
    Px_record = np.zeros((Nx,n-D))
    indexweight = Nx**np.arange(D)
    offset = (Nx**D - 1) // (Nx-1) + 1
    for i in range(D+1,n+1):
        context = x[i-D-1:i-1]
        leafindex = np.dot(context,indexweight)+offset
        xt = x[i-1]
        eta = (countTree[0:Nx-1,leafindex-1].T+0.5)/(countTree[Nx-1,leafindex-1]+0.5)
        # update the leaf
        countTree[xt,leafindex-1] = countTree[xt,leafindex-1] + 1
        node = np.floor((leafindex+Nx-2)/Nx)
        while ( node !=0 ):
            countTree, betaTree, eta = ctwupdate(countTree,betaTree, eta, node, xt,1/2)
            node = np.floor((node+Nx-2)/Nx)
        eta_sum = np.sum(eta)+1
        Px_record[:,i-D-1] = np.append(eta, [1]).T / eta_sum
    return Px_record


################################################################################

def compute_DI(X, Y, Nx, D, start_ratio):
    XY=X+Nx*Y
    n_data = len(X)

    pxy = ctwalgorithm(XY, Nx**2, D)
    px = ctwalgorithm(X, Nx, D)
    py = ctwalgorithm(Y, Nx, D)

    px_xy = np.zeros((Nx,n_data))

    for i_x in range(1, Nx+1):
        px_xy[i_x-1,:] = pxy[i_x-1,:]
        for j in range(2, Nx+1):
            px_xy[i_x-1,:] = px_xy[i_x-1,:] + pxy[i_x-1+(j-1)*Nx,:]

    temp = Nx*px_xy
    py_x_xy = np.divide(pxy, temp)

    # E1
    # temp_DI=  - np.log2( py[ Y[D,end-1]+ [1:Nx:end-Nx+1] ]) +  np.log2(pxy[XY[D,end-1]+[1:Nx^2:end-Nx^2+1]]) - np.log2(px_xy[X(D,end-1)+[1:Nx:end-Nx+1]])

    # E4
    temp_DI= np.zeros((1,px.shape[1]))
    for iy in range(1, Nx+1):
       for ix in range(1, Nx+1):
            tmp1 = pxy[ix+(iy-1)*Nx-1,:]
            tmp2 = np.log2( pxy[ix+(iy-1)*Nx-1,:] )
            tmp3 = np.multiply( py[iy-1,:], px_xy[ix-1,:] )
            temp_DI= temp_DI+ np.divide( np.multiply(tmp1, tmp2), tmp3 )

    DI = np.cumsum(temp_DI[np.floor(n_data*start_ratio),end-1])
    return DI


A = np.random.randint(0,2,10)
B = np.random.randint(0,2,10)
Nx = 2
D = 2
start_ratio = .1

print(compute_DI(A,B,Nx,D,start_ratio))






