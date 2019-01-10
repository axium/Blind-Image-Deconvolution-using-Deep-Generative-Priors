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

# algorithm constants
REGULARIZORS = [1.0, 0.5, 100.0, 0.001]
LEARNING_RATE = 0.005
RANDOM_RESTARTS = 10
NOISE_STD       = 0.01
STEPS           = 10000
IMAGE_RANGE     = [-1,1]
optimizer       = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)   
SAVE_PATH       = './results/CelebA/deblurring - alg2 - '+str(int(NOISE_STD*100)) + 'perc noise - ' +str(RANDOM_RESTARTS) + 'RR/deblurring_'
PLOT_LOSS       = True
SAVE_RESULTS    = True
# -----------------------------------------------------------------------

# loading blur test images
W = np.load(Blur_Path) 
BLUR_RES = W.shape[1]

# loading celeba test images
X_Orig = np.array([ imread(path) for path in glob(Orig_Path)])/255
X_Range = np.array([ imread(path) for path in glob(Range_Path)])/255
IMAGE_RES = X_Orig.shape[1]
CHANNELS = X_Orig.shape[-1]


# loading celeba generator
CelebAGen = CelebAGenerator()
CelebAGen.GenerateModel()
CelebAGen.LoadWeights()
CelebAGAN = CelebAGen.GetModels()
CelebAGAN.trainable = False
celeba_latent_dim = CelebAGen.latent_dim

# loading motion blur generator
BLURGen = MotionBlur()
BLURGen.GenerateModel()
BLURGen.LoadWeights()
blur_vae, blur_encoder, blur_decoder = BLURGen.GetModels()
blur_decoder.trainable = False
blur_latent_dim = BLURGen.latent_dim

# check if save dir exists, if not create a new one
try:
    os.stat(SAVE_PATH[:-11])
except:
    os.mkdir(SAVE_PATH[:-11])

# generating blurry images
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


# solving deconvolution using Algorithm 2
rr = Blurry_Images.shape[0]
zi_tf = tf.Variable(tf.random_normal(shape=([rr, celeba_latent_dim])), dtype = 'float32')
zk_tf = tf.Variable(tf.random_normal(shape=([rr, blur_latent_dim])), dtype = 'float32')
x_tf  = tf.Variable(tf.random_normal(mean = 0.5, stddev  = 0.01,shape=([rr, IMAGE_RES,IMAGE_RES,CHANNELS])))

x_G = CelebAGAN(zi_tf)
x_G = tf.reshape(x_G, shape=(rr,IMAGE_RES,IMAGE_RES,CHANNELS))
x_G = (x_G + 1)/2
y_fourier = tf.placeholder(shape=(rr, IMAGE_RES,IMAGE_RES,CHANNELS), dtype='complex64')


blur  = blur_decoder(zk_tf)
blur  = tf.reshape(blur, shape=(rr,BLUR_RES,BLUR_RES))
padding = np.int((IMAGE_RES -BLUR_RES)/2)
blur = tf.pad(blur, [[0,0], [padding,padding],[padding,padding]], 'CONSTANT')
blur_fourier = tf.fft2d( tf.cast(blur, dtype = 'complex64'))

# splitting tensors into 3 channels
y_fourier0 = y_fourier[:,:,:,0]; x_0 = x_tf[:,:,:,0]; x_G0 = x_G[:,:,:,0]
y_fourier1 = y_fourier[:,:,:,1]; x_1 = x_tf[:,:,:,1]; x_G1 = x_G[:,:,:,1]
y_fourier2 = y_fourier[:,:,:,2]; x_2 = x_tf[:,:,:,2]; x_G2 = x_G[:,:,:,2]

# 1st Channel Loss
x_0_fourier = tf.fft2d( tf.cast( x_0, dtype='complex64'))
loss_x0 = tf.reduce_mean( tf.square( tf.abs(y_fourier0 - x_0_fourier*blur_fourier) ), axis=[1,2])

x_Gi0_fourier = tf.fft2d( tf.cast( x_G0, dtype='complex64'))
loss_xG0 = tf.reduce_mean( tf.square( tf.abs(y_fourier0 - x_Gi0_fourier*blur_fourier) ), axis=[1,2])

# 2nd Channel Loss
x_1_fourier = tf.fft2d( tf.cast( x_1, dtype='complex64'))
loss_x1 = tf.reduce_mean( tf.square( tf.abs(y_fourier1 - x_1_fourier*blur_fourier) ), axis=[1,2])

x_Gi1_fourier = tf.fft2d( tf.cast( x_G1, dtype='complex64'))
loss_xG1 = tf.reduce_mean( tf.square( tf.abs(y_fourier1 - x_Gi1_fourier*blur_fourier) ), axis=[1,2])

# 3rd Channel Loss
x_2_fourier = tf.fft2d( tf.cast( x_2, dtype='complex64'))
loss_x2 = tf.reduce_mean( tf.square( tf.abs(y_fourier2 - x_2_fourier*blur_fourier) ), axis=[1,2])

x_Gi2_fourier = tf.fft2d( tf.cast( x_G2, dtype='complex64'))
loss_xG2 = tf.reduce_mean( tf.square( tf.abs(y_fourier2 - x_Gi2_fourier*blur_fourier) ), axis=[1,2])


Loss_xG_tf    = tf.constant(REGULARIZORS[0])*(loss_xG0 + loss_xG1 + loss_xG2)
Loss_x_tf     = tf.constant(REGULARIZORS[1])*(loss_x0 + loss_x1 + loss_x2)
x_minus_xG_tf = tf.constant(REGULARIZORS[2])*tf.reduce_mean( tf.square( tf.abs(x_tf - x_G)), axis=[1,2,3])
LossTV_tf     = tf.constant(REGULARIZORS[3])*tf.image.total_variation(x_tf)
TotalLoss_tf  = Loss_xG_tf + Loss_x_tf + x_minus_xG_tf + LossTV_tf

opt =  optimizer.minimize(TotalLoss_tf, var_list = [zi_tf, zk_tf, x_tf])
sess = K.get_session()
sess.run(tf.variables_initializer([zi_tf, zk_tf, x_tf]))
Losses = []

# running optimizer steps
for i in tqdm(range(STEPS), ascii=True, desc = 'Solving Deconv.'):
    losses = sess.run([opt, TotalLoss_tf, Loss_xG_tf, Loss_x_tf, x_minus_xG_tf], 
                      feed_dict = {y_fourier: Blurry_Images})
    Losses.append([loss for loss in losses[1:] ])
Losses = np.array(Losses)
zi_hat, zk_hat, x_hat = sess.run([zi_tf, zk_tf, x_tf])

tmp = []
for i in range(4):
    tmp.append( [loss[i] for loss in Losses])
Losses = tmp
TotalLoss, Loss_xG, Loss_x, x_minus_xG = Losses

# convergence plots 
if PLOT_LOSS:
    plt.figure(figsize=(10,5))
    plt.subplot(2,2,1)
    plt.plot(np.mean(TotalLoss, axis=1)); plt.title('Total Loss')
    plt.subplot(2,2,2)
    plt.plot(np.mean(Loss_x, axis=1)); plt.title('x Loss')
    plt.subplot(2,2,3)
    plt.plot(np.mean(Loss_xG, axis=1)); plt.title('xG Loss')
    plt.subplot(2,2,4)
    plt.plot(np.mean(x_minus_xG, axis=1)); plt.title('x - xG')
    plt.show()

# extracting best images from random restarts with minimum residual error
X_Hat = []
XG_Hat   = []
W_Hat = []
for i in range(len(X_Orig)):
    x_i      =      X_Orig[i]
    zi_hat_i = zi_hat[i*RANDOM_RESTARTS:(i+1)*RANDOM_RESTARTS]
    zk_hat_i = zk_hat[i*RANDOM_RESTARTS:(i+1)*RANDOM_RESTARTS]
    x_hat_i    = x_hat[i*RANDOM_RESTARTS:(i+1)*RANDOM_RESTARTS]
    w_hat_i    = blur_decoder.predict(zk_hat_i)[:,:,:,0]
    x_hat_i      = np.clip(x_hat_i, 0, 1)
    loss_i       = [ComputeResidual(Y_np[i], x, w) for x,w in zip(x_hat_i,w_hat_i)]
    min_loss_loc = np.argmin(loss_i)
    
    zi_hat_recov = zi_hat_i[min_loss_loc].reshape([1,celeba_latent_dim])
    zk_hat_recov = zk_hat_i[min_loss_loc].reshape([1,blur_latent_dim])
    x_hat_recov  = x_hat_i[min_loss_loc] 
    w_hat = blur_decoder.predict(zk_hat_recov).reshape(BLUR_RES,BLUR_RES)
    xg_hat = CelebAGAN.predict(zi_hat_recov).reshape(IMAGE_RES,IMAGE_RES,CHANNELS)
    X_Hat.append(x_hat_recov); W_Hat.append(w_hat); XG_Hat.append(xg_hat)
X_Hat = np.array(X_Hat)
W_Hat = np.array(W_Hat)
XG_Hat = np.array(XG_Hat)

# normalizing images
X_Hat = np.clip(X_Hat, 0,1)
XG_Hat = (XG_Hat + 1)/2
    
# calculating psnr and ssim --- in paper both PSNR and SSIM where computed
# using matlab
PSNR = []; SSIM = []
for x, x_pred in zip(X_Orig, X_Hat):
    psnr = compare_psnr(x, x_pred.astype('float64'))
    ssim = compare_ssim(x, x_pred.astype('float64'), multichannel=True)
    PSNR.append(psnr); SSIM.append(ssim)
print("PSNR = ", np.mean(PSNR))
print("SSIM = ", np.mean(SSIM))


# saving results
Max = 10**len(str(len(X_Orig)-1))
if SAVE_RESULTS:
    for i in range(len(X_Orig)):
        Save_Results(path = SAVE_PATH + str(i+Max)[1:], 
                         x_np = None, 
                         w_np = None,
                         y_np = Y_np[i], 
                         y_np_range = None , 
                         x_hat_test = X_Hat[i], 
                         w_hat_test = W_Hat[i], 
                         x_range = None, 
                         x_hat_range = XG_Hat[i], 
                         w_hat_range = None, clip=True)