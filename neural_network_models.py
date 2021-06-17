import os
import numpy as np
import pickle
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F

#the pytorch module subclassing classes for projector and generator models
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
#         self.dense0 = nn.Linear(1024, 4*4*256)
#         self.bn0 = nn.BatchNorm1d(4*4*256)
        self.tconv0 = nn.ConvTranspose2d(256, 256, 4, 1, padding=0, bias=False)
        self.bn0 = nn.BatchNorm2d(256)

        self.tconv1 = nn.ConvTranspose2d(256, 192, 4, 2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(192)

        self.tconv2 = nn.ConvTranspose2d(192, 128, 4, 2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.tconv3 = nn.ConvTranspose2d(128, 64 , 4, 2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.tconv4 = nn.ConvTranspose2d(64, 32, 4, 2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(32)
        
        self.tconv5 = nn.ConvTranspose2d(32, 16 , 4, 2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(16)
        
        self.tconv6 = nn.ConvTranspose2d(16, 1, 4, 2, padding=1, bias=False)
        
        

    def forward(self, x): 
        x = F.relu(self.bn0(self.tconv0(x)))
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        x = F.relu(self.bn4(self.tconv4(x)))
        x = F.relu(self.bn5(self.tconv5(x)))
        img = torch.tanh(self.tconv6(x))

        return img

    
class Projector(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 8, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        
        self.conv2 = nn.Conv2d(8, 16, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        
        self.conv3 = nn.Conv2d(16, 32, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        
        self.conv4 = nn.Conv2d(32, 64, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.conv5= nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        
        self.conv6 = nn.Conv2d(128, 192, 4, 2, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(192)
        
        self.conv7 = nn.Conv2d(192, 256, 4, 1, 0, bias=False)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        latent_est = self.conv7(x)
        
        return latent_est

#Info-GAN based denosiser class for the ADMM step 2 update
class GenBasedDenoiser():
    def __init__(self, model_loc, g_model,proj_model):
        self.checkpoint1 = torch.load(os.path.join(model_loc,'checkpoint_model_aconv4ks_mod_latent_C_9_3_0055_d0_000001_g0_000001.pickle'))            
        g_model.load_state_dict(self.checkpoint1['netG'])
        self.g_model = g_model.eval()
        
        self.checkpoint2 = torch.load(os.path.join(model_loc, 'proj_trained_params_alconv_lat_mod_gen_imgs.pth'))
        proj_model.load_state_dict(self.checkpoint2['netP'])
        self.proj_model = proj_model.eval()
        
    def denoise(self, x):
        x = torch.reshape(torch.Tensor(x), (-1,1,256,256))
        z = self.proj_model(x)
        recon_img = self.g_model(z)
        return recon_img.detach()[0,0,:,:]
    
