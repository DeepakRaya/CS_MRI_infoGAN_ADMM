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
        
        self.dense0 = nn.Linear(1024, 4*4*256)
        self.bn0 = nn.BatchNorm1d(4*4*256)

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
        x = F.relu(self.dense0(x))
        x = torch.reshape(x, (-1,256,4,4))
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
        
        self.fc1 = nn.Linear(65536, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096,2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048,1024)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        latent_est = self.fc3(x)
        
        return latent_est
    
#Info-GAN based denosiser class for the ADMM step 2 update
class GenBasedDenoiser():
    def __init__(self, model_loc, g_model,proj_model):
        self.checkpoint1 =torch.load(os.path.join(model_loc,'checkpoint_model_4_5_d0_0001_g0_0001.pickle'))                              
        g_model.load_state_dict(self.checkpoint1['netG'])
        self.g_model = g_model.eval()
        self.checkpoint2 = torch.load(os.path.join(model_loc, 'proj_trained_params_4_2_1.pth'))
        proj_model.load_state_dict(self.checkpoint2['netP'])
        self.proj_model = proj_model.eval()
    def denoise(self, x):
        x = torch.reshape(torch.Tensor(x), (-1,65536))
        z = self.proj_model(x)
        recon_img = self.g_model(z)
        return recon_img.detach()[0,0,:,:]
    
