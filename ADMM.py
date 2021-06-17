
import numpy as np
#import scipy.sparse.linalg as linalg
from copy import deepcopy as deepcopy
from undersampling2 import usam_data
from stacked_optim import stack_ls_optim
import torch

class PnPADMM(object):
    def __init__(self, M, N, algo_param=None):
        '''
        (M, N): M - number of measurements, N - original dimension
        algo_param: parameter dictionary containing:
            "rho": parameter to control the constraint x = z
            "x0": starting point all the algorithm
            "tol": allowance for number of non-loss-decreasing iterations
            "maxiter": maximum number of iterations
            "callback": call back function, called after each solving iteration
                        must take input in the order of: x, z, u, loss
        '''
        self.shape = (M, N)
        if algo_param is None:
            self.algo_param = {}
        else:
            self.algo_param = deepcopy(algo_param)
            
        if "rho" not in self.algo_param:
            self.algo_param["rho"] = 1e2
            
        if "x0" not in self.algo_param:
            self.algo_param["x0"] = np.random.randn((N,N))
            
        if "tol" not in self.algo_param:
            self.algo_param["tol"] = 50
        
        if "maxiter" not in self.algo_param:
            self.algo_param["maxiter"] = 100
            
        if "callback" not in self.algo_param:
            self.algo_param["callback"] = None
                 
        
    def solve(self, y, A_mask, A, Denoiser):
        '''
        Use the plug-and-play ADMM algorithm for compressive sensing recovery
        '''
        #P = np.linalg.inv(A.T.dot(A) + self.algo_param["rho"]*np.eye(self.shape[1]))
        P = np.linalg.inv(A.T.dot(A) + self.algo_param["rho"]*np.eye(self.shape[1]))
        
        # Initial start
        loss_func = lambda x: np.sum(np.square(y - usam_data(x, A_mask))) # define obj.                                                                                    func.
        
#         loss = loss_func(self.algo_param["x0"])
#         loss_star = loss
        
        x_f = usam_data(self.algo_param["x0"],A_mask)
        z_f = x_f
        u_f = np.zeros_like(x_f,dtype = np.complex128)
        
#         x = self.algo_param["x0"]
#         z = np.zeros_like(x)
#         u = np.zeros_like(x)
        
        k = 0
        tol = self.algo_param["tol"]
        loss_record = np.array([])
        z_record = []
        x_record = []
        callback_res = []
        
        # Start iterations
        while k < self.algo_param["maxiter"]:
            # least square step
            x_f = stack_ls_optim(P, A, y, z_f, u_f, self.algo_param["rho"])
            
            x = np.real(np.fft.ifft2(x_f))
            # denoising step
            z = np.array(Denoiser.denoise(x))
            z_f = np.fft.fft2(z)
            # dual variable update
            u_f += x_f - z_f
            # monitor the loss
            loss =  loss_func(z)
#             if loss < loss_star:
#                 loss_star = loss
#             else:
#                 tol -= 1
                
            loss_record = np.append(loss_record, loss)
            # record all the denoised signals
            z_record.append(z)
            x_record.append(x)
            # callback functions
            if self.algo_param["callback"] is not None:
                callback_res.append(self.algo_param["callback"](x, z, u, loss))
                
            k += 1
        print(k)
        print(np.argmin(loss_record))
          
     
        x_star = z_record[np.argmin(loss_record)]
        xx_star = x_record[np.argmin(loss_record)]
       
    
        return x_star, xx_star, loss_record