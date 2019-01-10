import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, Input, Lambda, Reshape, Flatten
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D
import keras.backend as K
from keras import initializers

class MotionBlur():
    def __init__(self):
        self.latent_dim = 50        # Dimension of Latent Representation
        self.Encode = None
        self.Decoder = None
        self.VAE = None
        self.weights_path = './model weights/motionblur.h5'
        
    def GenerateModel(self):
        
        # Encoder input layer
        input_ = Input(shape=(28,28,1))

        # Encoder hidden layers
        encoder_hidden1 = Conv2D(filters= 20, kernel_size=2, strides = (1,1), padding='valid', activation='relu')(input_)
        encoder_hidden2 = MaxPooling2D(pool_size = (2,2), strides= (2,2))(encoder_hidden1)
        encoder_hidden3 = Conv2D(filters = 20, kernel_size=2, strides = (1,1), padding='valid',activation='relu' )(encoder_hidden2)
        encoder_hidden4 = MaxPooling2D( pool_size = (2,2), strides= (2,2))(encoder_hidden3)
        encoder_hidden5 = Flatten()(encoder_hidden4)

        # Latent Represenatation Distribution, P(z)
        z_mean = Dense(self.latent_dim, activation='linear', 
                                  kernel_initializer= initializers.he_normal(seed=None))(encoder_hidden5)
        z_std_sq_log = Dense(self.latent_dim, activation='linear', 
                                  kernel_initializer= initializers.he_normal(seed=None))(encoder_hidden5)

        # Sampling z from P(z)
        def blur_sample_z(args):
            mu, std_sq_log = args
            epsilon = K.random_normal(shape=(K.shape(mu)[0], self.latent_dim), mean=0., stddev=1.)
            z = mu + epsilon * K.sqrt( K.exp(std_sq_log)) 
            return z

        z = Lambda(blur_sample_z)([z_mean, z_std_sq_log])


        # Decoder/Generator hidden layers
        decoded_hidden1 = Dense(K.int_shape(encoder_hidden5)[1], activation='relu', kernel_initializer= initializers.he_normal())(z)
        decoded_hidden2 = Reshape(K.int_shape(encoder_hidden4)[1:])(decoded_hidden1)
        decoder_hidden3 = UpSampling2D(size=(2,2))(decoded_hidden2)
        decoder_hidden4 = Conv2DTranspose(filters = 20, kernel_size=2, strides = (1,1), padding='valid',activation='relu')(decoder_hidden3)
        decoder_hidden5 = UpSampling2D(size=(2,2))(decoder_hidden4)
        decoder_hidden6 = Conv2DTranspose(filters = 20, kernel_size=2, strides = (1,1), padding='valid',activation='relu')(decoder_hidden5)

        output = Conv2DTranspose(filters = 1, kernel_size=2, strides = (1,1), padding='valid',activation='relu')(decoder_hidden6)

        # VAE MODEL
        vae = Model(input_, output)
        encoder =  Model(inputs = input_, outputs = [z_mean, z_std_sq_log])

        blur_input_decoder = Input( shape=(self.latent_dim,))
        blur_hidden1_decoder = vae.layers[9](blur_input_decoder)
        blur_hidden2_decoder = vae.layers[10](blur_hidden1_decoder)
        blur_hidden3_decoder = vae.layers[11](blur_hidden2_decoder)
        blur_hidden4_decoder = vae.layers[12](blur_hidden3_decoder)
        blur_hidden5_decoder = vae.layers[13](blur_hidden4_decoder)
        blur_hidden6_decoder = vae.layers[14](blur_hidden5_decoder)
        blur_output_decoder = vae.layers[15](blur_hidden6_decoder)

        decoder = Model(blur_input_decoder, blur_output_decoder)


        self.VAE = vae
        self.Encoder = encoder
        self.Decoder  = decoder

        
    def LoadWeights(self):
        self.VAE.load_weights(self.weights_path)

    def GetModels(self):
        return self.VAE, self.Encoder, self.Decoder


if __name__ == "__main__":
    BLURGen = MotionBlur()
    BLURGen.GenerateModel()
    BLURGen.weights_path = '../model weights/motionblur.h5'
    BLURGen.LoadWeights()
    blur_vae, blur_encoder, blur_decoder = BLURGen.GetModels()
    
    n_samples = 10
    len_z = BLURGen.latent_dim
    z = np.random.normal(0,1,size=(n_samples*n_samples ,len_z))
    sampled = blur_decoder.predict(z)
    
    k = 0
    for i in range(n_samples):
        for j in range(n_samples):
            blur = sampled[k]
            blur = blur/blur.max()
            blur = blur[:,:,0]
            plt.subplot(n_samples,n_samples,k+1)
            plt.imshow(blur, cmap="gray")
            plt.axis("Off")
            k=k+1
    plt.show()