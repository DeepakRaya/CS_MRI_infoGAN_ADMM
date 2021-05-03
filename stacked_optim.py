import numpy as np
def stack_ls_optim(P, A, y, z, w, rho):
    x = np.zeros_like(y)
    for i in range(y.shape[1]):
        q = P.dot(A.T.dot(y[:,i]))
        x[:,i] = q + rho*P.dot(z[:,i]-w[:,i])
    return x
        