import numpy as np
from skimage.io import imsave, imread
import keras.backend as K
from Utils import *
from glob import glob
from generators.CelebAGenerator import *
from generators.ShoeGenerator import *
from generators.SVHNGenerator import *
K.set_learning_phase(0)
import matplotlib.pyplot as plt

'''
            FIND RANGE IMAGES FOR TEST IMAGES
            
1. FOR SVHN   ---> STEPS = 6,000,  STEP SIZE = 0.01
2. FOR CelebA ---> STEPS = 10,000, STEP SIZE = 0.001
3. FOR Shoes  ---> STEPS = 10,000, STEP SIZE = 0.001
'''
# constants
DATASET = 'SVHN' # or "CelebA" or "Shoes"
STEPS = 6000
STEP_SIZE = 0.01
RANDOM_RESTARTS = 10

# paths
ORIG_PATH = './results/' + DATASET + '/Original Images/*.png'
RANGE_PATH = './results/' + DATASET + '/Range Images/'

# loading test images
X_Orig = []
for path in glob(ORIG_PATH):
    image = imread(path).astype('float32')/255
    if (DATASET == 'CelebA' or DATASET == 'Shoes'):
        image = (image - 0.5)*2
    X_Orig.append(image)
X_Orig = np.array(X_Orig)


# loading generator
if DATASET == 'CelebA':
    celeba_gan = CelebAGenerator()
    celeba_gan.GenerateModel()
    celeba_gan.LoadWeights()
    Generator = celeba_gan.GetModels()
    latent_dim = celeba_gan.latent_dim
    
if DATASET == 'Shoes':
    shoes_gan = ShoeGenerator()
    shoes_gan.GenerateModel()
    shoes_gan.LoadWeights()
    Generator = shoes_gan.GetModels()
    latent_dim = shoes_gan.latent_dim
    
if DATASET == 'SVHN':
    svhn_gan = SVHNGenerator()
    svhn_gan.GenerateModel()
    svhn_gan.LoadWeights()
    _, _, Generator = svhn_gan.GetModels()
    latent_dim = svhn_gan.latent_dim
    

# solving for range images
X_Range = Find_in_Range_Parallel(X_Orig, rr  = RANDOM_RESTARTS,  
                                 decoder =  Generator,
                                 latent_dim =  latent_dim, steps = STEPS, 
                                 step_size = STEP_SIZE)
if (DATASET == 'CelebA' or DATASET == 'Shoes'):
    X_Range = (X_Range + 1)/2

# saving range images
Max = 10**len(str(len(X_Orig)))
for i in range(len(X_Orig)):
    imsave(RANGE_PATH + 'x_range_' + str(Max+i)[1:] + '.png', X_Range[i] )
