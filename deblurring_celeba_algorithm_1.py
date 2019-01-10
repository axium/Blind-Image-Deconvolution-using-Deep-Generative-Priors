import tensorflow as tf
import keras.backend as K
import numpy as np
from Utils import *
from generators.MotionBlurGenerator import *
from generators.CelebAGenerator import *
K.set_learning_phase(0)
from glob import glob
import os


# paths
Orig_Path       = './results/CelebA/Original Images/*.png'
Range_Path      = './results/CelebA/Range Images/*.png'
Blur_Path       = './results/CelebA/Original Blurs/Test Blurs.npy'

# constants
REGULARIZORS = [0.01 , 0.01]
RANDOM_RESTARTS = 10
NOISE_STD       = 0.01
STEPS           = 10000
IMAGE_RANGE = [-1,1]

def step_size(t):
    return 0.01 * np.exp( - t / 1000 )

SAVE_PATH       = './results/CelebA/deblurring - alg1 - ' +str(int(NOISE_STD*100)) + 'perc noise - ' +str(RANDOM_RESTARTS) + 'RR/deblurring_'
# -----------------------------------------------------------------------

# loading test blur images
W = np.load(Blur_Path) 
BLUR_RES = W.shape[1]

# loading test celeba images
X_Orig = np.array([ imread(path) for path in glob(Orig_Path)])/255
X_Range = np.array([ imread(path) for path in glob(Range_Path)])/255
IMAGE_RES = X_Orig.shape[1]
CHANNELS = X_Orig.shape[-1]

# loading celeba generator
CelebAGen = CelebAGenerator()
CelebAGen.GenerateModel()
CelebAGen.LoadWeights()
CelebAGAN = CelebAGen.GetModels()
celeba_latent_dim = CelebAGen.latent_dim

# loading motion blur generator
BLURGen = MotionBlur()
BLURGen.GenerateModel()
BLURGen.LoadWeights()
blur_vae, blur_encoder, blur_decoder = BLURGen.GetModels()
blur_latent_dim = BLURGen.latent_dim

# check if save dir exists, if not create a new one
try:
    os.stat(SAVE_PATH[:-11])
except:
    os.mkdir(SAVE_PATH[:-11])

# generating blurry images from test
Y_np = []
Blurry_Images = []
for i in tqdm(range(len(X_Orig)), ascii=True, desc ='Gen-Test-Blurry'):
    x_np = X_Orig[i]
    w_np = W[i]
    y_np, y_f = GenerateBlurry(x_np, w_np, noise_std = NOISE_STD )
    Y_np.append(y_np)
    for _ in range(RANDOM_RESTARTS):
        Blurry_Images.append(y_f)

Y_np = np.array(Y_np)
Blurry_Images = np.array(Blurry_Images)

# generating blurry images from range
Blurry_Images_range = []
Y_np_range = []
for i in tqdm(range(len(X_Orig)), ascii=True, desc ='Gen-Range-Blurry'):
    y_np, y_f = GenerateBlurry(X_Range[i], W[i], noise_std = NOISE_STD )
    Y_np_range.append(y_np)
    for _ in range(RANDOM_RESTARTS):
        Blurry_Images_range.append(y_f)

Y_np_range = np.array(Y_np_range)
Blurry_Images_range = np.array(Blurry_Images_range)


# alternating gradient descent for test images
image_gradients, blur_gradients, get_loss = Generate_Gradient_Functions(rr = Blurry_Images.shape[0],
                                                                        reg = REGULARIZORS, image_range = IMAGE_RANGE,
                                                                        decoder = CelebAGAN, blur_decoder = blur_decoder,
                                                                        image_res = IMAGE_RES, blur_res = BLUR_RES,
                                                                        channels = CHANNELS)
m_hat, h_hat, Loss = Optimize_Parallel(blurry_fourier = Blurry_Images, stepsize=step_size,steps = STEPS,
                                      image_grad = image_gradients , blur_grad = blur_gradients, 
                                      getloss = get_loss, latent_image_dim = celeba_latent_dim , latent_blur_dim = blur_latent_dim)
X_hat_test = []
W_hat_test = []
for i in range(len(X_Orig)):
    m_hat_i = m_hat[i*RANDOM_RESTARTS:(i+1)*RANDOM_RESTARTS]
    h_hat_i = h_hat[i*RANDOM_RESTARTS:(i+1)*RANDOM_RESTARTS]
    Loss_i  =  Loss[i*RANDOM_RESTARTS:(i+1)*RANDOM_RESTARTS]
    x_hat_test, w_hat_test, loss_last_iter_test = Get_Min_Loss(Loss_i, m_hat_i, h_hat_i, decoder = CelebAGAN, blur_decoder = blur_decoder,
                                                               latent_image_dim = celeba_latent_dim, latent_blur_dim = blur_latent_dim,  print_grad=False)  
    X_hat_test.append(x_hat_test)
    W_hat_test.append(w_hat_test)

X_hat_test = np.array(X_hat_test)
W_hat_test = np.array(W_hat_test)

# alternating gradient descent for range images
m_hat, h_hat, Loss = Optimize_Parallel(blurry_fourier = Blurry_Images_range, stepsize=step_size,steps = STEPS,
                                      image_grad = image_gradients , blur_grad = blur_gradients, 
                                      getloss = get_loss, latent_image_dim = celeba_latent_dim , latent_blur_dim = blur_latent_dim)
X_hat_range = []
W_hat_range = []
for i in range(len(X_Orig)):
    m_hat_i = m_hat[i*RANDOM_RESTARTS:(i+1)*RANDOM_RESTARTS]
    h_hat_i = h_hat[i*RANDOM_RESTARTS:(i+1)*RANDOM_RESTARTS]
    Loss_i  = Loss[i*RANDOM_RESTARTS:(i+1)*RANDOM_RESTARTS]
    x_hat_range, w_hat_range, loss_last_iter_range = Get_Min_Loss(Loss_i, m_hat_i, h_hat_i, decoder = CelebAGAN, blur_decoder = blur_decoder,
                                                                  latent_image_dim = celeba_latent_dim, latent_blur_dim = blur_latent_dim, print_grad=False)  
    X_hat_range.append(x_hat_range)
    W_hat_range.append(w_hat_range)

X_hat_range = np.array(X_hat_range)
W_hat_range = np.array(W_hat_range)

X_hat_test = (X_hat_test + 1)/2
X_hat_range = (X_hat_range + 1)/2
Max = 10**len(str(len(X_Orig)-1))

# saving results
for i in range(len(X_Orig)):
    Save_Results(path = SAVE_PATH + str(i+Max)[1:], 
                     x_np = None, 
                     w_np = None,
                     y_np = Y_np[i], 
                     y_np_range = Y_np_range[i] , 
                     x_hat_test = X_hat_test[i], 
                     w_hat_test = W_hat_test[i], 
                     x_range = None, 
                     x_hat_range = X_hat_range[i], 
                     w_hat_range = W_hat_range[i], clip=True)