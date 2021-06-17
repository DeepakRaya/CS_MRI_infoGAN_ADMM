import numpy as np
def usam_data(data_tensor, mask):
    k_space = np.fft.fft2(data_tensor[:,:])
#     k_space  = np.fft.fftshift(fourier)
    #N = k_space.shape[0]
    subsam_k_space = np.multiply(mask, k_space)#undersampling in k-space
#     zfr = np.fft.ifft2(np.fft.ifftshift(subsam_k_space))
    return subsam_k_space