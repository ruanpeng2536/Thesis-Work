import os
import cv2
import keras
import random
import scipy
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage import measure
from keras.layers import Input, Dense, Lambda
from keras.layers.merge import concatenate as concat
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from ssim import SSIM
from ssim.utils import get_gaussian_kernel
from PIL import Image, ImageOps, ImageEnhance 
from scipy.stats import wasserstein_distance
from skimage.transform import resize
from math import log2, log10
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

# ========================= GET RESULTS =========================

n_z = 1000 

# specify resized image sizes
height = 28 #2**10
width = 28 #2**10

def load_NPZ_data(file_dir):
    # returns: train_images , train_labels 
    data = np.load(file_dir)
    return data['arr_0'], data['arr_1']

font_folder = 'MY_TOTAL_CVAE_z1000'
flag_Naive = True

for method_number in range(5): # each method
    for s_step in range(5): # each percent step
        each_step = [0.01,0.05,0.1,0.25,0.5] # 1%, 5%, 10%, 25% , and 50% each step
        if (flag_Naive and s_step == 4): # 50% (Naive)
            step = each_step[s_step] 
            flag_Naive = False
        elif (flag_Naive != True and s_step == 4):
            continue
        else:
            step = each_step[s_step] 

        sides = int(100/(100*step)) + 1 # number of latent space walks (=1 because it also count the last one)
        path = 'TOTAL_fonts'

        #                 0     1       2       3       4     5      6     7      8      9       10     11     12     13
        # typeOfMethod = ['mse','rmse','psnr','rmse_sw','uqi','ssim','scc','rase','sam','msssim','vifp','psnrb','cs','cw-ssim']
        typeOfMethod = ['mse','uqi','ssim','vifp','cw-ssim']
        methodFolder = ['MSE','UQI','SSIM','VIFP','CW-SSIM']
        ind = method_number

        first_font = 9
        second_font = 18


        # Create directory
        if (step == 0.5):
            pair_font = font_folder+ '/' + str(first_font+1)+'+'+str(second_font+1)
            dirName = pair_font+'/Synthesized_Letters_z'+str(n_z)+'_'+typeOfMethod[ind]+'_'+str(first_font+1)+'+'+str(second_font+1)+'_L'+str(step)
        else:
            pair_font = font_folder + '/' +str(first_font+1)+'+'+str(second_font+1) + '/'
            dirName = pair_font+'/'+methodFolder[method_number]+'/Synthesized_Letters_z'+str(n_z)+'_'+typeOfMethod[ind]+'_'+str(first_font+1)+'+'+str(second_font+1)+'_L'+str(step)

        class_labels = list(range(1, 27)) # [1,26]
        model = tf.keras.models.load_model('myAllFonts_model.h5')
        npz_dir = dirName+'/'+'z'+str(n_z)+'_'+typeOfMethod[ind]+'_'+str(first_font+1)+'+'+str(second_font+1)+'_L'+str(step)

        first_images, first_labels = load_NPZ_data(npz_dir+'_FIRST.npz') # shape: (260,28,28) 10 each alphabet
        syn_images, syn_labels = load_NPZ_data(npz_dir+'_SYN.npz')       # shape: (260,) 10 each alphabet
        second_images, second_labels = load_NPZ_data(npz_dir+'_SECOND.npz')

        pre_first_images = np.reshape(first_images, (-1,28,28,1))
        predict_first = model.predict(pre_first_images) 
        predict_first_result_index = np.argmax(predict_first, 1)

        pre_syn_images = np.reshape(syn_images, (-1,28,28,1))
        predict_syn = model.predict(pre_syn_images) 
        predict_syn_result_index = np.argmax(predict_syn, 1)

        pre_second_images = np.reshape(second_images, (-1,28,28,1))
        predict_second = model.predict(pre_second_images) 
        predict_second_result_index = np.argmax(predict_second, 1)

        

        # print(f'pre_first_images: {pre_first_images.shape}, predict_first: {predict_first.shape}')
        print()
        print(f'_________ {npz_dir} _________FIRST__SYN__SECOND_________')
        acc_first = accuracy_score(first_labels, predict_first_result_index)
        acc_syn = accuracy_score(syn_labels, predict_syn_result_index)
        acc_second = accuracy_score(second_labels, predict_second_result_index)
        print(acc_first)
        print(acc_syn)
        print(acc_second)
        
        
        
plt.imshow(first_images[0])
plt.show()
        

        