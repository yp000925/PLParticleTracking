'''
This manuscript is used to generate the simulated DH figure under different noise level (Poisson noise)
'''
import matplotlib.pyplot as plt
import numpy as np
import random


# Define the size of the mosaic pattern
SIZE = 8

# Generate a random value for each pixel in the mosaic pattern
mosaic = [[random.randint(0, 255) for _ in range(SIZE)] for _ in range(SIZE)]

# Plot the mosaic pattern as an image
plt.imshow(mosaic, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.show()


#%%
from utils.utilis import generate_otf_torch
import torch
from PIL import Image
from torch.nn import functional as F

wavelength=532.3e-9
deltax=1.12e-6
deltay=1.12e-6
distance=1.065e-3
nx=256
ny=256
# pad_size = (512, 512)
obj = Image.open('DH.png').convert('L').resize((256, 256))
obj = np.array(obj)/ np.array(obj).max()
obj = torch.tensor(obj)
# obj = F.pad(torch.tensor(obj), (pad_size[0]-obj.shape[0], pad_size[0]-obj.shape[0],pad_size[1]-obj.shape[0], pad_size[1]-obj.shape[1]), mode='constant', value=0)


otf = generate_otf_torch(wavelength, obj.shape[0], obj.shape[1], deltax, deltay, distance)

fs= torch.multiply(torch.fft.fft2(obj), otf)
holo = torch.fft.ifft2(fs)
amplitude = holo.abs()/holo.abs().max()
plt.imshow(amplitude, cmap='gray', vmin=0, vmax=1)


#%% add Poisson noise
ALPHA = 20
y_n = np.random.poisson(np.maximum(ALPHA*amplitude,0))
plt.imshow(y_n.tolist(),cmap='gray')
plt.imsave('y_n.png', y_n.tolist(),cmap='gray')