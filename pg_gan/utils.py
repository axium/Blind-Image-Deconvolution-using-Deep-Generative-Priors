
import copy
import numpy as np
import torch
from torch.autograd import Variable
from skimage.io import imsave

def scale_image(image, as_uint8=True):
    img = copy.deepcopy(image).astype("float")
    img -= img.min()
    img /= img.max()
    if as_uint8:
        img *= 255
        return img.astype(np.uint8)
    else:
        return img


def Generate_Blurry(x_np, w_np, noise_std = 0.01, clip=True):
    x_np  = np.transpose(x_np, [2,0,1])
    x_np  = np.expand_dims(x_np, axis=0)
    w_np = np.expand_dims(w_np, axis=0)
    w_np  = np.expand_dims(w_np, axis=0)
    x_torch, w_torch = torch.FloatTensor(x_np), torch.FloatTensor(w_np)
    
    x_torch1 = Variable(x_torch)[:,:1,:,:]
    x_torch2 = Variable(x_torch)[:,1:2,:,:]
    x_torch3 = Variable(x_torch)[:,2:3,:,:]

    w_torch = Variable(w_torch)
    
    # Convolving Blur Kernel
    blurry1 = torch.nn.functional.conv2d(x_torch1, w_torch, padding=1)
    blurry2 = torch.nn.functional.conv2d(x_torch2, w_torch, padding=1)
    blurry3 = torch.nn.functional.conv2d(x_torch3, w_torch, padding=1)
    
    blurry  = np.concatenate([blurry1.data.numpy(), blurry2.data.numpy() , blurry3.data.numpy()], axis=1)
    blurry  = np.transpose(blurry, [0,2,3,1])[0]
    blurry  = blurry + np.random.normal(0,noise_std, blurry.shape)
    blurry  = np.clip(blurry, 0,1)
    return blurry 

def Convolve_Torch(x_torch, w_torch):
    x_torch1 = x_torch[:,:1,:,:]
    x_torch2 = x_torch[:,1:2,:,:]
    x_torch3 = x_torch[:,2:3,:,:]
    
    blurry1 = torch.nn.functional.conv2d(x_torch1, w_torch, padding=1)
    blurry2 = torch.nn.functional.conv2d(x_torch2, w_torch, padding=1)
    blurry3 = torch.nn.functional.conv2d(x_torch3, w_torch, padding=1)
    blurry = torch.cat((blurry1, blurry2, blurry3), dim=1)
    return blurry

    
def Optimize(optimizer, y, z_i, z_k, CelebA_Gen, blur_Decoder,regularizers, return_loss = True ):
   
    w_i = blur_Decoder(z_k)
    x_i = CelebA_Gen(z_i)
    x_i = x_i - x_i.min(); x_i = x_i / x_i.max()
    y_pred = Convolve_Torch(x_i, w_i)
    
    optimizer.zero_grad()
    Loss = torch.dist(y_pred, y, p=2)  # Computing 2-Norm  
    Loss_with_Regularizors = Loss + regularizers[0]*torch.norm(z_i, p=2)  + regularizers[1]*torch.norm(z_k, p=2)
    Loss_with_Regularizors.backward()
    optimizer.step()
    if return_loss:
        return z_i, z_k, Loss.data.cpu().numpy().tolist()
    else:
        return z_i, z_k
    
    
def Save_Results(path, x_np, y_np, x_hat, clip=True, save_as_np = False):
    if clip:
        x_np = np.clip(x_np, 0,1)
        y_np = np.clip(y_np, 0,1)
        x_hat = np.clip(x_hat, 0,1)
        
    if save_as_np == False:
        imsave(path+'_x_orig.png', (x_np*255).astype('uint8'))
        imsave(path+'_y.png',      (y_np*255).astype('uint8'))
        imsave(path+'_x_hat.png',  (x_hat*255).astype('uint8'))
    if save_as_np == True:
        np.save(path+'_x_orig.png',x_np )
        np.save(path+'_y.png', y_np)
        np.save(path+'_x_hat.png',x_hat )
