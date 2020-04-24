import os
import keras
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Input, Dense, Lambda
from keras.layers.merge import concatenate as concat
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from skimage import measure
warnings.filterwarnings('ignore')

def load_data_from_file(file_dir):
    # returns: train_images , train_labels 
    data = np.load(file_dir)
    return data['arr_0'], data['arr_1'], data['arr_2']

def sample_z(args):
    mu, l_sigma = args
    eps = K.random_normal(shape=(m, n_z), mean=0., stddev=1.)
    return mu + K.exp(l_sigma / 2) * eps

def vae_loss(y_true, y_pred):
    print(y_true)
    print(y_pred)
    recon = K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)
    kl = 0.5 * K.sum(K.exp(l_sigma) + K.square(mu) - 1. - l_sigma, axis=-1)
    return recon + kl

def KL_loss(y_true, y_pred):
    return(0.5 * K.sum(K.exp(l_sigma) + K.square(mu) - 1. - l_sigma, axis=1))

def recon_loss(y_true, y_pred):
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

def construct_numvec(digit, z = None):
    out = np.zeros((1, n_z + n_y))
    out[:, digit + n_z] = 1.
    if z is None:
        return(out)
    else:
        for i in range(len(z)):
            out[:,i] = z[i]
        return(out)

def collect_data(path):
    path_dir = os.listdir(path)
    acc_images = []
    acc_labels = []
    for font in path_dir:
        x,y,_ = load_data_from_file(path+'/'+font)
        acc_images.append(x)
        acc_labels.append(y)
    return np.array(acc_images), np.array(acc_labels)
    
font_folder = 'MY_TOTAL_CVAE_z1000'
print('### DATA LOADED!! ###')
# path = 'mini_font/MINI_PAIR_Fonts_a-yummy-apology+blah-blah-bang.npz'

path = 'TOTAL_fonts'
# images, labels = collect_data(path)
# images = images.reshape((-1,28,28))
# labels = labels.reshape((-1))



''' OLD PAIR OF FONTS'''
# x_train_test, y_train_test, z_train_test = load_data_from_file(path)
# x_train = x_train_test.reshape(*x_train_test.shape[:1], -1) #/ 255
# print(x_train_test.shape)
# print(y_train_test.shape)

''' OLD PAIR OF FONTS'''

x_train_test, y_train_test = collect_data(path)
x_train_test = x_train_test.reshape((-1,28,28))
y_train_test = y_train_test.reshape((-1))
x_train = x_train_test.reshape(*x_train_test.shape[:1], -1) #/ 255
print(x_train.shape)
print(y_train_test.shape)


test_data_size =  26000 #2600 # MINI_font test -> '2496' #67250# 10000  #x_train_test.shape[0]//6 # 1:6 ratio
train_data_size = 130000 #20800# MINI_font train -> '13056' #400000 
train_img = []
train_label = []
test_img = []
test_label = []
m = 8 # batch size
n_z = 1000 # latent space size
encoder_dim1 = 512 # dim of encoder hidden layer
decoder_dim = 512 # dim of decoder hidden layer
decoder_out_dim = 784 # dim of decoder output layer
activ = 'relu'
optim = Adam(lr=0.001)
n_epoch = 200
number_of_class = 27

wantTest = False
for idx in range(x_train_test.shape[0]):
    if (idx < test_data_size):
        test_img.append(x_train_test[idx])
        test_label.append(y_train_test[idx])
    else:
        wantTest = True
    if(wantTest and idx < (train_data_size+test_data_size)):
        train_img.append(x_train_test[idx])
        train_label.append(y_train_test[idx])

# print(f'train_img: {train_img.shape}, test_img: {test_img.shape}')
# print(f'X_train: {y_train.shape}, X_test: {y_test.shape}')

# plt.imshow(train_img[0])
# plt.show()
train_img = np.asarray(train_img, dtype=np.float32)/ 255.0
train_label = np.asarray(train_label, dtype=np.float32)
test_img = np.asarray(test_img, dtype=np.float32)/ 255.0
test_label = np.asarray(test_label, dtype=np.float32)
tl = to_categorical(test_label)
# print(f'tl = {tl[0]}')
(X_train, Y_train), (X_test, Y_test) = (train_img, train_label), (test_img, test_label)
n_pixels = np.prod(X_train.shape[1:])

X_train = X_train.reshape((len(X_train), n_pixels))
X_test = X_test.reshape((len(X_test), n_pixels))
y_test = to_categorical(Y_test,number_of_class)
y_train = to_categorical(Y_train,number_of_class)

# print(f'X_train: {X_train.shape}, X_test: {X_test.shape}')
# print(f'X_train: {y_train.shape}, X_test: {y_test.shape}')

n_x = X_train.shape[1]
n_y = y_train.shape[1]


X = Input(shape=(n_x,))
label = Input(shape=(n_y,))

inputs = concat([X, label])

encoder_h = Dense(encoder_dim1, activation=activ)(inputs)
mu = Dense(n_z, activation='linear')(encoder_h)
l_sigma = Dense(n_z, activation='linear')(encoder_h)

z = Lambda(sample_z, output_shape = (n_z, ))([mu, l_sigma]) # Sampling latent space

# merge latent space with label
zc = concat([z, label])

decoder_hidden = Dense(decoder_dim, activation=activ)
decoder_out = Dense(decoder_out_dim, activation='sigmoid')
h_p = decoder_hidden(zc)
outputs = decoder_out(h_p)

cvae = Model([X, label], outputs)
encoder = Model([X, label], mu)

d_in = Input(shape=(n_z+n_y,))
d_h = decoder_hidden(d_in)
d_out = decoder_out(d_h)
decoder = Model(d_in, d_out)

cvae.compile(optimizer=optim, loss=vae_loss, metrics = [KL_loss, recon_loss])


# compile and fit
cvae_hist = cvae.fit([X_train, y_train], X_train, verbose = 1, batch_size=m, epochs=n_epoch,
							validation_data = ([X_test, y_test], X_test),
							callbacks = [EarlyStopping(patience = 50)])

print(cvae_hist.history.keys())

plt.plot(cvae_hist.history['KL_loss'])
plt.plot(cvae_hist.history['recon_loss'])
plt.plot(cvae_hist.history['loss'])
plt.title('model loss')
plt.ylabel('training loss')
plt.xlabel('epoch')
plt.legend(['KL_loss', 'recon_loss','vae_loss'], loc='upper right')
plt.show()

plt.plot(cvae_hist.history['val_KL_loss'])
plt.plot(cvae_hist.history['val_recon_loss'])
plt.plot(cvae_hist.history['val_loss'])

plt.title('model validation loss')
plt.ylabel('validation loss')
plt.xlabel('epoch')
plt.legend(['val_KL_loss','val_recon_loss','val_loss'], loc='upper right')
plt.show()


plt.imshow(X_train[0].reshape(28, 28), cmap = 'gray')
plt.show()
print(Y_train[0])

# save models
encoder.save(font_folder+"/encoder_model_z1000.h5")
decoder.save(font_folder+"/decoder_model_z1000.h5")
encoded_X0 = encoder.predict([X_train[0].reshape((1, 784)), y_train[0].reshape((1, number_of_class))])
print(encoded_X0)


# t-NSE graph
z_train = encoder.predict([X_train, y_train])
encodings= np.asarray(z_train)
encodings = encodings.reshape(X_train.shape[0], n_z)
plt.figure(figsize=(7, 7))
sc = plt.scatter(encodings[:, 0], encodings[:, 1], c=Y_train, cmap=plt.cm.jet)
plt.colorbar(sc)
plt.show()

sample_3 = construct_numvec(5) # control output here
plt.figure(figsize=(3, 3))
plt.imshow(decoder.predict(sample_3).reshape(28,28), cmap = 'gray')
plt.show()





