import torch
import numpy as np
import torch.nn.functional as F

def generate_gaussian_window(win_size, sigma):
    g_win = np.random.normal(win_size//2, sigma, (win_size,win_size))
    return torch.reshape(torch.Tensor(g_win/np.sum(g_win)),(-1,win_size,win_size))

def ssim(img1,img2,val_range,sigma, window_size=11, size_average=True):
    L = val_range # L is the dynamic range of the pixel values 
    pad = window_size // 2
    sha = img1.shape
    img1 = torch.reshape(torch.Tensor(img1),(-1,)+sha).unsqueeze_(0) 
    img2 = torch.reshape(torch.Tensor(img2),(-1,)+sha).unsqueeze_(0) 


    _, channels, height, width = img1.size()
    print(img1.size())

    window = generate_gaussian_window(window_size, sigma).unsqueeze_(0) 

    # calculating the mu parameter (locally) for both images using a gaussian filter 
    # calculates the luminosity params
    mu1 = F.conv2d(img1, window,padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window,padding=pad, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2 
    mu12 = mu1 * mu2

    # now we calculate the sigma square parameter
    # Sigma deals with the contrast component 
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 =  F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12

    # Some constants for stability 
    C1 = (0.01 ) ** 2  # NOTE: Removed L from here (ref PT implementation)
    C2 = (0.03 ) ** 2 

    contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    contrast_metric = torch.mean(contrast_metric)

    numerator1 = 2 * mu12 + C1  
    numerator2 = 2 * sigma12 + C2
    denominator1 = mu1_sq + mu2_sq + C1 
    denominator2 = sigma1_sq + sigma2_sq + C2

    ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)

    if size_average:
        ret = ssim_score.mean() 
    else: 
        ret = ssim_score.mean(1).mean(1).mean(1)

#     if full:
#         return ret, contrast_metric

    return ret
    