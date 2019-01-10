import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
import keras.backend as K
from keras.layers import AveragePooling2D
from tqdm import tqdm
from skimage.measure import compare_psnr, compare_ssim



# remove this function 
def Blur_Pad(w):
    return np.lib.pad(w, 2,mode='constant', constant_values = 0)

# a function to generate blurry images y: y = x * w + n 
def GenerateBlurry(x, w, noise_std):
    '''
    x = Image To Be BLURRED
    w = BLUR KERNEL
    '''
    channels = x.shape[-1]
    padding = np.int((x.shape[0] - w.shape[0])/2)
    w = np.lib.pad(w, padding, mode='constant', constant_values = 0)
    
    # applying blur kernel on each channel in freq. domain
    y_f = np.zeros(x.shape, dtype=complex)
    for i in range(channels):
        y_f[:,:,i] = np.fft.fft2(x[:,:,i]) * np.fft.fft2(w)
        
    # converting to spatial domain
    y = np.zeros(x.shape)
    for i in range(channels):
        y[:,:,i] = np.fft.fftshift( np.fft.ifft2(y_f[:,:,i]).real)
    
    # adding noise
    noise = np.random.normal(0,noise_std, size = y.shape)
    y = y + noise
    y = np.clip(y,0,1)
    
    # converting noisy image back to freq. domain
    for i in range(channels):
        y_f[:,:,i] = np.fft.fft2(np.fft.fftshift(y[:,:,i]))
    return y, y_f

# a function to compute residual error: || y - x * w || 
def ComputeResidual(y, x, w):
    y_pred, _ = GenerateBlurry(x, w, noise_std=0)
    return np.linalg.norm(y_pred - y)

# a function to find images in range of a generator -- old unoptimized version
def Find_in_Range(x_np, latent_dim, decoder, steps = 1000, step_size = 0.01, print_and_show = True):
    find_span_of = x_np # IMAGE TO FIND THE CLOSEST SPAN OF.

    original_image = K.placeholder(shape = x_np.shape)
    image_reshape = K.reshape( decoder.output, shape = x_np.shape )
    loss_span = K.sum( K.square( K.abs(original_image - image_reshape) ))
    span_gradients = K.function( [decoder.input, original_image], K.gradients(loss_span, [decoder.input]))
    
    m_hat = np.random.normal(0,1,size=[1,latent_dim])
    for i in tqdm(range(steps)):
        grad_m = np.array(span_gradients([m_hat, find_span_of])).reshape([1, latent_dim])
        m_hat = m_hat - step_size*grad_m
    x_range_decoder = decoder.predict(m_hat).reshape(x_np.shape)
    if print_and_show:
        print('PSNR  = ', compare_psnr(find_span_of.astype('float64'), x_range_decoder.astype('float64') ))
        print('SSIM  = ', compare_ssim(find_span_of.astype('float64'), x_range_decoder.astype('float64'), multichannel=True ))
        f, ax = plt.subplots(1,2, figsize=(5,5))
        ax[0].imshow(find_span_of); ax[0].axis('Off')
        ax[1].imshow(x_range_decoder); ax[1].axis('Off')
        plt.show()
        return x_range_decoder
    else:
        return x_range_decoder
    
# a function to find images in range of a generator -- new optimized version
def Find_in_Range_Parallel(X_np, rr, decoder, latent_dim, steps = 6000, step_size = 0.01, print_and_show = True):
    
    X_rr = []
    for x in X_np:
        for i in range(rr):
            X_rr.append(x)
    X_rr = np.array(X_rr)
    n_images = X_rr.shape[0]
    original_image = K.placeholder(shape= X_rr.shape)
    image_reshape = K.reshape( decoder.output, shape = X_rr.shape )
    loss_span = K.sum( K.square( original_image - image_reshape ), axis=[1,2,3])
    span_gradients = K.function( [decoder.input, original_image], K.gradients(loss_span, [decoder.input]))
    m_hat = np.random.normal(0,1,size=[n_images,latent_dim])
    for i in tqdm(range(steps), ascii=True, desc = 'Finding Range'):
        grad_m = np.array(span_gradients([m_hat, X_rr])).reshape([n_images, latent_dim])
        m_hat = m_hat - step_size*grad_m
    X_recov = decoder.predict(m_hat).reshape(X_rr.shape)
    X_Range = []
    for i in range(len(X_np)):
        x_recov = X_recov[i*rr:(i+1)*rr]
        error = [ np.linalg.norm(x - X_np[i]) for x in x_recov ]
        idx = np.argmin( error)
        X_Range.append(x_recov[idx])
    X_Range = np.array(X_Range)
    if print_and_show:
        PSNR =[]; SSIM = [];
        for img, rng in zip(X_np.astype('float32'), X_Range.astype('float32')):
            PSNR.append(compare_psnr(img, rng)); SSIM.append(compare_ssim(img, rng, multichannel=True))
        print('\n\n Average PSNR = %0.3f .... Average SSIM = %0.3f \n\n' %(np.mean(PSNR), np.mean(SSIM)))
            
    return X_Range

# a function solve deconvolution - new optimized version
def Optimize_Parallel(blurry_fourier, stepsize, steps, image_grad, blur_grad, getloss, latent_image_dim, latent_blur_dim):
    
    Loss = []
    h_hat = []
    m_hat = []
    rr = blurry_fourier.shape[0]

        # random intialization of m_hat and h_hat
    m_hat_i = np.random.normal( 0,   1, [rr,latent_image_dim])
    h_hat_i = np.random.normal( 0,  1, [rr,latent_blur_dim])
        
    Loss_i = np.array([])   
    for j in tqdm(range(steps), ascii=True, desc = 'Solving Deconvolution'):
        # Updating h_hat and m_hat
        delta_h = np.array(blur_grad([m_hat_i, h_hat_i, blurry_fourier])).reshape([rr,latent_blur_dim])
        delta_h = delta_h / (np.sqrt(np.mean(np.square(delta_h), axis=-1)) + 1e-5).reshape(rr,1)
        h_hat_i = h_hat_i - stepsize(j) * delta_h

        delta_m = np.array(image_grad([m_hat_i, h_hat_i, blurry_fourier])).reshape([rr, latent_image_dim])
        delta_m = delta_m /  ( np.sqrt(np.mean(np.square(delta_m), axis=-1)) + 1e-5).reshape(rr,1)
        m_hat_i = m_hat_i -  stepsize(j) * delta_m


        # Calculating Loss
        loss = getloss([m_hat_i, h_hat_i, blurry_fourier])[0]
        Loss.append(loss)
    h_hat.append(h_hat_i)
    m_hat.append(m_hat_i)

    return np.array(m_hat)[0], np.array(h_hat)[0], np.array(Loss).T



def Get_Min_Loss(Loss, m_hat, h_hat, decoder, blur_decoder, latent_image_dim, latent_blur_dim,  m_grad=None, h_grad=None, print_grad=False):
    min_loss_loc = Loss[:,-1].argmin()
    h_hat_recovered = h_hat[min_loss_loc]
    m_hat_recovered = m_hat[min_loss_loc]

    h_hat_recovered = h_hat_recovered.reshape([1,latent_blur_dim])
    m_hat_recovered = m_hat_recovered.reshape([1,latent_image_dim])
    if print_grad:
        print('The gradient of ciphar10 at the last iterations = ', m_grad[min_loss_loc,-1])
        print('The gradient of blur at the last iterations = ', h_grad[min_loss_loc,-1])
        
    w_hat = blur_decoder.predict(h_hat_recovered)[0,:,:,0]
    x_hat = decoder.predict(m_hat_recovered)[0]
    return x_hat, w_hat, np.sqrt(Loss[min_loss_loc,-1])
    

# a function used to get functions that will return gradients
def Generate_Gradient_Functions(rr, reg, decoder, image_res, 
                                blur_decoder, blur_res, image_range, channels):
    '''
    rr     = Random Restarts
    reg    = Regularizors list (2-dimensional)
    res    = Low-resolution
    factor = low_resolution_factor , image_range (2-dimensional with Low and Max Value)
    '''
    # Splitting Tensors into 3 channels
    if channels == 3:
        y = K.placeholder(shape=(rr,image_res,image_res,channels), dtype = 'complex64')
        padding = np.int((image_res - blur_res)/2)
        # Reshaping and Padding Tensors
        image_reshaped = K.reshape(decoder.output, shape=[rr, image_res, image_res, channels] )
        image_reshaped = image_reshaped - image_range[0]
        image_reshaped = image_reshaped / (image_range[1] - image_range[0])
        blur_reshaped = K.reshape(blur_decoder.output, shape=[rr,blur_res,blur_res])
        blur_reshaped = K.tf.pad(blur_reshaped, [ [0,0],[padding,padding],[padding,padding]], 'CONSTANT')
        blur_fourier = K.tf.fft2d( K.tf.cast( blur_reshaped, dtype='complex64'))
        
        y0 = y[:,:,:,0]; image0 = image_reshaped[:,:,:,0]
        y1 = y[:,:,:,1]; image1 = image_reshaped[:,:,:,1]
        y2 = y[:,:,:,2]; image2 = image_reshaped[:,:,:,2]

        # 1st Channel Loss
        image0_fourier = K.tf.fft2d( K.tf.cast( image0, dtype='complex64'))
        predicted0_fourier = image0_fourier * blur_fourier
        blind_loss0 = K.sum( K.square( K.abs(y0 - predicted0_fourier)), axis=[1,2] )
        
        # 2nd Channel Loss
        image1_fourier = K.tf.fft2d( K.tf.cast( image1, dtype='complex64'))
        predicted1_fourier = image1_fourier * blur_fourier
        blind_loss1 = K.sum( K.square( K.abs(y1 - predicted1_fourier)), axis=[1,2] )

        # 3rd Channel Loss
        image2_fourier = K.tf.fft2d( K.tf.cast( image2, dtype='complex64'))
        predicted2_fourier = image2_fourier * blur_fourier
        blind_loss2 = K.sum( K.square( K.abs(y2 - predicted2_fourier )), axis=[1,2] )

        # Residual Loss
        residual_error = blind_loss0 + blind_loss1 + blind_loss2
        blind_loss = residual_error + reg[0]*K.sum(K.square(blur_decoder.input), axis=1) + reg[1]*K.sum(K.square(decoder.input), axis=1)

        # Functions to get gradients of blind_loss w.r.t Blur and Mnist Decoder Inputs.
        blur_gradients = K.function( [decoder.input, blur_decoder.input, y], K.gradients(blind_loss, [blur_decoder.input]))
        image_gradients = K.function( [decoder.input, blur_decoder.input, y], K.gradients(blind_loss, [decoder.input]))
        get_loss = K.function([decoder.input, blur_decoder.input, y], [residual_error])
    
        return image_gradients, blur_gradients, get_loss
    
    if channels == 1:
        y = K.placeholder(shape=(rr, low_res,low_res), dtype = 'complex64')
        image_reshaped = K.reshape(decoder.output, shape = [rr, image_res,image_res])
        image_reshaped = image_reshaped - image_range[0]
        image_reshaped = image_reshaped / (image_range[1] - image_range[0])
        blur_reshaped = K.reshape(blur_decoder.output, shape=[rr, blur_res,blur_res])
        image_fourier = K.tf.fft2d( K.tf.cast(image_reshaped, dtype='complex64'))
        blur_fourier = K.tf.fft2d( K.tf.cast( blur_reshaped, dtype='complex64'))
        predicted_fourier = image_fourier*blur_fourier
        predicted = K.tf.ifft2d(predicted_fourier)
        predicted = K.tf.cast(predicted, dtype='float32')
        predicted = K.reshape(predicted, shape = [rr,image_res,image_res,channels])
        residual_loss = K.sum( K.square(y - predicted), axis=[1,2] ) 
        blind_loss = residual_loss + reg[0]*K.sum(K.square(decoder.input), axis=1) + reg[1]*K.sum(K.square(blur_decoder.input), axis=1)

        blur_gradients = K.function( [decoder.input, blur_decoder.input, y], K.gradients(blind_loss, [blur_decoder.input]))
        image_gradients = K.function( [decoder.input, blur_decoder.input, y], K.gradients(blind_loss, [decoder.input]))
        get_loss = K.function([decoder.input, blur_decoder.input, y], [residual_loss])
        return image_gradients, blur_gradients, get_loss 


# a function to plot results
def Plot_Results(Loss, x_recon_error, w_recon_error,  m_grad, h_grad):
    # Plotting Loss of each random restart
    plt.figure(figsize=(15,15))
    
    for i in range(Loss.shape[0]):
        plt.subplot(321)
        plt.plot(10 * np.log10(Loss[i]))
    plt.title('Blind Deconvolution Loss (db)',size=15)
    
    if x_recon_error is not None:
        plt.subplot(322)
        for i in range(x_recon_error.shape[0]):
            plt.plot(x_recon_error[i])
        plt.title('x_recon_error',size=15)

    if w_recon_error is not None:
        plt.subplot(312)
        for i in range(w_recon_error.shape[0]):
            plt.plot(w_recon_error[i])
        plt.title('w_recon_error',size=15)
        
    if h_grad is not None:
        plt.subplot(325)
        for i in range(h_grad.shape[0]):
            plt.plot( 10*np.log10(h_grad[i]))
        plt.title('Gradient of h',size=15)

    if m_grad is not None:
        plt.subplot(326)
        for i in range(m_grad.shape[0]):
            plt.plot( 10*np.log10(m_grad[i]))
        plt.title('Gradient of m',size=15)

    plt.show()

# a function to save results for deblurring
def Save_Results(path, x_np, w_np, y_np, y_np_range, x_hat_test, w_hat_test, x_range,x_hat_range, w_hat_range, clip=True):
    if clip:
        if x_np        is not None: x_np = np.clip(x_np, 0,1)
        if w_np        is not None: w_np = np.clip(w_np, 0,1)
        if y_np        is not None: y_np = np.clip(y_np, 0,1)
        if x_hat_test  is not None: x_hat_test = np.clip(x_hat_test, 0,1)
        if w_hat_test  is not None: w_hat_test = np.clip(w_hat_test, 0,1)
        if x_range     is not None: x_range = np.clip(x_range, 0,1)
        if x_hat_range is not None: x_hat_range = np.clip(x_hat_range, 0,1)
        if w_hat_range is not None: w_hat_range = np.clip(w_hat_range, 0,1)
        if y_np_range  is not None: y_np_range  = np.clip(y_np_range, 0,1)
            
    if x_np        is not None:  imsave(path+'_x_orig.png', (x_np*255).astype('uint8'))
    if w_np        is not None:  imsave(path+'_w_orig.png', (w_np/w_np.max() * 255).astype('uint8'))
    if y_np        is not None:  imsave(path+'_y_from_test.png',      (y_np*255).astype('uint8'))
    if y_np_range  is not None:  imsave(path+'_y_from_range.png',     (y_np_range*255).astype('uint8'))
    if x_hat_test  is not None:  imsave(path+'_x_hat_from_test.png',  (x_hat_test*255).astype('uint8'))
    if w_hat_test  is not None:  imsave(path+'_w_hat_from_test.png',  (w_hat_test/w_hat_test.max() * 255).astype('uint8'))
    if x_range     is not None:  imsave(path+'_x_range.png', (x_range*255).astype('uint8'))
    if x_hat_range is not None:  imsave(path+'_x_hat_from_range.png', (x_hat_range*255).astype('uint8'))
    if w_hat_range is not None:  imsave(path+'_w_hat_from_range.png', (w_hat_range/w_hat_range.max() * 255).astype('uint8'))


