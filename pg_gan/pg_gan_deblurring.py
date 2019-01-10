import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import torch
from torch.autograd import Variable
from pggan_model import Generator
from blur_model import blur
from utils import *
import warnings
from tqdm import tqdm
from glob import glob
from natsort import natsorted
warnings.filterwarnings("ignore")


'''
Blind Image Deconvolution using PG-GAN as generative model G_i for images. 
Sometimes it may get stuck in local minima, and may produce results different 
from paper. Simply re-run the code or increase the number of random restarts.
 
Also note that blurs are selected randomly from the test-set so some deviations
are expected from the results in paper.
'''

# constants
REGULARIZORS    = [0.001, 0.001]
STEPS           = 2000
STEP_SIZE       = 0.01
NOISE_STD       = 0.01 # std of 0.01 translates to 1 perc noise
RANDOM_RESTARTS = 10
IMAGE_IDX       = 2 # which image to use from folder: results/orig/

# setting up generators
CelebAGen = Generator()
CelebAGen.load_state_dict(torch.load('./weights/pggan_generator.pth'))
CelebAGen.cuda()
CelebAGen.eval()

Blur_Gen = torch.load("./weights/blur_generator.pt")
Blur_Gen = Blur_Gen.cuda()
Blur_Gen.eval()

# loading blur test set
blur_test = np.load("data/blur_test.npy")
idx       = np.random.randint(0,len(blur_test))
blur      = blur_test[idx].squeeze()


# loading saved sampled images
orig_path = natsorted(glob("./results/orig/*"))
x_true = [imread(p) for p in orig_path]
x_true = [scale_image(x, as_uint8=False) for x in x_true]
x_true = np.array(x_true)[IMAGE_IDX]

# generating blurry image
y = Generate_Blurry(x_true, blur, noise_std = NOISE_STD)

# solving deconvolution

y_torch = np.expand_dims(np.transpose(y, [2,0,1]), axis=0)
y_torch = Variable(torch.FloatTensor(y_torch).cuda(), requires_grad = False)

Loss = []
zk = []
zi = []
for j in range(RANDOM_RESTARTS): # Looping over Random Restarts
    # Initializing z_i and z_k
    z_k = Variable(torch.randn(1,50).cuda(),requires_grad=True)
    z_i = Variable(torch.randn(1,512,1,1).cuda(),requires_grad=True)
    Loss_j = []
    optimizer = torch.optim.Adam([z_i, z_k], lr = STEP_SIZE)
    # running gradient steps
    for i in tqdm(range(STEPS)): 
        z_i, z_k, loss = Optimize(optimizer, y_torch,z_i,z_k,CelebA_Gen=CelebAGen,blur_Decoder=Blur_Gen, 
                                  regularizers=REGULARIZORS,return_loss=True )
        Loss_j.append(loss)
        
    zi.append(z_i.data.cpu().numpy())
    zk.append(z_k.data.cpu().numpy())
    Loss.append(Loss_j)
Loss = np.array(Loss) 

# plotting losses  
plt.figure(figsize=(15,5))
for i in range(RANDOM_RESTARTS):
    plt.plot(np.log10(Loss[i]))
plt.title('Loss for ' + str(RANDOM_RESTARTS) + ' Random Restarts (db)')
plt.show()

# selecting best from random restarts with minimum residual loss
arg_min = Loss[:,-1].argmin()
zi_hat = zi[arg_min]
zk_hat = zk[arg_min]

# recovering kernel
zk_hat = torch.Tensor(zk_hat).cuda()
zk_hat = Variable(zk_hat, requires_grad=False)
w_hat  = Blur_Gen(zk_hat).data.cpu().numpy().squeeze()

# recovering image
zi_hat = torch.Tensor(zi_hat).cuda()
zi_hat = Variable(zi_hat, requires_grad=False)
x_hat = CelebAGen(zi_hat).data.cpu().numpy()
x_hat = scale_image(x_hat, as_uint8=False)
x_hat = np.transpose(x_hat, [0,2,3,1]).squeeze()

# displaying results
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(y); plt.axis("Off"); plt.title("Blurry")
plt.subplot(1,3,2)
plt.imshow(x_hat); plt.axis("Off"); plt.title("Deblurred")
plt.subplot(1,3,3)
plt.imshow(x_true); plt.axis("Off"); plt.title("Original")
plt.show()

# saving results properly in results folder
i = 0
saved_paths = natsorted(glob("./results/*x_orig.png"))
for p in saved_paths:
    path = "./results/pg_gan_%d_noiseperc_%d"%(i,int(NOISE_STD*100))
    if path in p:
        i=i+1
    else:
        break
path = "./results/pg_gan_%d_noiseperc_%d"%(i,int(NOISE_STD*100))
Save_Results(path, x_true, y, x_hat, clip=True, save_as_np = False)
