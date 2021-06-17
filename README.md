# CS_MRI_infoGAN_ADMM
## Load_data.py: 
For converting and loading the data from neuroimaging format ".nii.gz" to pickel format, normalize the images in the range -1 to 1.
## CS_MRI_ADMM_pytorch.ipynb:
Notebook contains the info-GAN model, Projector Model and their training routines.
## neural_network_models.py:
The saved info-GAN generator and projector network are loaded in this module and composition of generator and projector network is performed, which is used as a denoiser in the second step of ADMM algorithm.
## ADMM.py:
The Plug and Play ADMM solver algorithm program.
## undersampling2.py:
Performs the undersampling in k-space of MRI image input and returns a Zero filled reconstruction.
## stacked_optim.py:
Performs the first least squares like step of ADMM in a stacked column wise fashion.
## SSIM.py:
For calculating the SSIM score between original and reconstructed images, using 11x11 gaussian window.
## Experiment.ipynb:
Reconstrction experiments and resutls compilation notebook.
