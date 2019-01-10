import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv2D, BatchNormalization, Activation, Dense, Conv2DTranspose, Input, Lambda, Reshape, Flatten, UpSampling2D, MaxPooling2D
from keras.models import Model, Sequential
import keras.backend as K
from keras import initializers


class ShoeGenerator():

    def __init__(self):
        self.latent_dim = 100        # Dimension of Latent Representation
        self.GAN = None
        self.weights_path = './model weights/shoes.h5'

        
    def GenerateModel(self):
        gf_dim = 64
        gan = Sequential()
        gan.add(Dense(8192, use_bias = True, bias_initializer='zeros', input_dim=100))
        gan.add(Reshape([4,4,gf_dim*8]))
        gan.add(BatchNormalization(epsilon = 1e-5,momentum = 0.9,scale = True)) 
        gan.add(Activation('relu'))
        gan.add(Conv2DTranspose(gf_dim*4, 5, strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = initializers.random_normal(stddev=0.02), bias_initializer = 'zeros')) #see in channel value and std_value for random normal
        gan.add(BatchNormalization(epsilon = 1e-5,momentum = 0.9,scale = True)) 
        gan.add(Activation('relu'))
        gan.add(Conv2DTranspose(gf_dim*2, 5, strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = initializers.random_normal(stddev=0.02), bias_initializer = 'zeros')) #see in channel value and std_value for random normal
        gan.add(BatchNormalization(epsilon = 1e-5,momentum = 0.9,scale = True)) 
        gan.add(Activation('relu'))
        gan.add(Conv2DTranspose(gf_dim*1, 5, strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = initializers.random_normal(stddev=0.02), bias_initializer = 'zeros')) #see in channel value and std_value for random normal
        gan.add(BatchNormalization(epsilon = 1e-5,momentum = 0.9,scale = True)) 
        gan.add(Activation('relu'))
        gan.add(Conv2DTranspose(3, 5, strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = initializers.random_normal(stddev=0.02), bias_initializer = 'zeros')) #see in channel value and std_value for random normal
        gan.add(Activation('tanh')) 

        self.GAN = gan


    def LoadWeights(self):
        self.GAN.load_weights(self.weights_path)

    def GetModels(self):
        return self.GAN

if __name__ == '__main__':
    Gen = ShoeGenerator()
    Gen.GenerateModel()
    Gen.weights_path = '../model weights/shoes.h5'
    Gen.LoadWeights()
    gan = Gen.GetModels()
    
    n_samples = 10
    len_z = Gen.latent_dim
    z = np.random.normal(0,1,size=(n_samples*n_samples ,len_z))
    sampled = gan.predict(z)
    sampled = (sampled+1)/2
    
    k = 0
    for i in range(n_samples):
        for j in range(n_samples):
            img = sampled[k]
            plt.subplot(n_samples,n_samples,k+1)
            plt.imshow(img)
            plt.axis("Off")
            k=k+1
    plt.show()