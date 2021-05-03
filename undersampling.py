import numpy as np
def usam_data(data_tensor, mask):
    fourier = np.fft.fft2(data_tensor[:,:])
    cen_fourier  = np.fft.fftshift(fourier)
    subsam_fourier = np.multiply(cen_fourier,mask) #undersampling in k-space
    uncen_fourier = np.fft.ifftshift(subsam_fourier)
    zro_image = np.fft.ifft2(uncen_fourier) #zero-filled reconstruction
  
    return zro_image 