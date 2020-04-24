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


# sewar pakages
from sewar.full_ref import mse
from sewar.full_ref import rmse
from sewar.full_ref import psnr
from sewar.full_ref import rmse_sw
from sewar.full_ref import uqi
from sewar.full_ref import ssim
from sewar.full_ref import scc
from sewar.full_ref import rase
from sewar.full_ref import sam
from sewar.full_ref import msssim
from sewar.full_ref import vifp
# from sewar.full_ref import psnrb

warnings.filterwarnings('ignore')

# ========================= GET RESULTS =========================

n_z = 100 

# specify resized image sizes
height = 28 #2**10
width = 28 #2**10

def load_data_from_file(file_dir):
    # returns: train_images , train_labels 
    data = np.load(file_dir)
    return data['arr_0'], data['arr_1'], data['arr_2']

def construct_numvec(letter, z = None):
    out = np.zeros((1, n_z + n_y))
    out[:, letter + n_z] = 1.
    if z is None:
        return(out)
    else:
        for i in range(len(z)):
            out[:,i] = z[i] 
        return(out)

# to get latent values by percentage (linear interpolation)
def latent_walk(z_train,first,second,p):
    temp = []
    for idx in range(n_z):
        temp.append((z_train[first][idx] * p) + (z_train[second][idx] * (1-p)))
    return temp

def _plot_loss(history,letter):
    hist = pd.DataFrame(history)
    plt.figure(figsize=(20,5))
    plt.suptitle(letter)
    for colnm in hist.columns:
        plt.plot(hist[colnm],label=colnm)
    plt.legend()
    plt.ylabel("value")
    plt.xlabel("step")
    plt.show()

# def mse(x, y):
#     return np.linalg.norm(x - y)

def _compute_bef(im, block_size=8):
	"""Calculates Blocking Effect Factor (BEF) for a given grayscale/one channel image
	C. Yim and A. C. Bovik, "Quality Assessment of Deblocked Images," in IEEE Transactions on Image Processing,
		vol. 20, no. 1, pp. 88-98, Jan. 2011.
	:param im: input image (numpy ndarray)
	:param block_size: Size of the block over which DCT was performed during compression
	:return: float -- bef.
	"""
	if len(im.shape) == 3:
		height, width, channels = im.shape
	elif len(im.shape) == 2:
		height, width = im.shape
		channels = 1
	else:
		raise ValueError("Not a 1-channel/3-channel grayscale image")

	if channels > 1:
		raise ValueError("Not for color images")

	h = np.array(range(0, width - 1))
	h_b = np.array(range(block_size - 1, width - 1, block_size))
	h_bc = np.array(list(set(h).symmetric_difference(h_b)))

	v = np.array(range(0, height - 1))
	v_b = np.array(range(block_size - 1, height - 1, block_size))
	v_bc = np.array(list(set(v).symmetric_difference(v_b)))

	d_b = 0
	d_bc = 0

	# h_b for loop
	for i in list(h_b):
		diff = im[:, i] - im[:, i+1]
		d_b += np.sum(np.square(diff))

	# h_bc for loop
	for i in list(h_bc):
		diff = im[:, i] - im[:, i+1]
		d_bc += np.sum(np.square(diff))

	# v_b for loop
	for j in list(v_b):
		diff = im[j, :] - im[j+1, :]
		d_b += np.sum(np.square(diff))

	# V_bc for loop
	for j in list(v_bc):
		diff = im[j, :] - im[j+1, :]
		d_bc += np.sum(np.square(diff))

	# N code
	n_hb = height * (width/block_size) - 1
	n_hbc = (height * (width - 1)) - n_hb
	n_vb = width * (height/block_size) - 1
	n_vbc = (width * (height - 1)) - n_vb

	# D code
	d_b /= (n_hb + n_vb)
	d_bc /= (n_hbc + n_vbc)

	# Log
	if d_b > d_bc:
		t = log2(block_size)/log2(min(height, width))
	else:
		t = 0

	# BEF
	bef = t*(d_b - d_bc)

	return bef

def psnrb(GT, P):
	"""Calculates PSNR with Blocking Effect Factor for a given pair of images (PSNR-B)
	:param GT: first (original) input image in YCbCr format or Grayscale.
	:param P: second (corrected) input image in YCbCr format or Grayscale..
	:return: float -- psnr_b.
	"""
	if len(GT.shape) == 3:
		GT = GT[:, :, 0]

	if len(P.shape) == 3:
		P = P[:, :, 0]

	imdff = np.double(GT) - np.double(P)

	mse = np.mean(np.square(imdff.flatten()))
	bef = _compute_bef(P)
	mse_b = mse + bef

	if np.amax(P) > 2:
		psnr_b = 10 * log10(255**2/mse_b)
	else:
		psnr_b = 10 * log10(1/mse_b)

	return psnr_b

def ssd(img1, img2):
    """Computing the sum of squared differences (SSD) between two images."""
    return np.sum((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32))**2)

def crr(img1, img2):
    """Computing the Cross-Correlation (CRR) between two images."""
    return np.sum((np.array(img1, dtype=np.float32) * np.array(img2, dtype=np.float32))**2)

def hd(img1, img2):
    return scipy.spatial.distance.hamming(np.ravel(img1), np.ravel(img2))

def earth_movers_distance(img1, img2):
    '''
    Measure the Earth Mover's distance between two images
    '''
    img_1 = get_img(img1, norm_exposure=True)
    img_2 = get_img(img2, norm_exposure=True)
    hist_a = get_histogram(img_1)
    hist_b = get_histogram(img_2)
    return wasserstein_distance(hist_a, hist_b)

def get_histogram(img):
    '''
    Get the histogram of an image. For an 8-bit, grayscale image, the
    histogram will be a 256 unit vector in which the nth value indicates
    the percent of the pixels in the image with the given darkness level.
    The histogram's values sum to 1.
    '''
    # plt.imshow(img)
    # plt.show()
    # plt.imshow(img.astype(int))
    # plt.show()
    h, w = img.shape
    hist = [0.0] * 256
    for i in range(h):
        for j in range(w):
            hist[int(img[i, j])] += 1
    return np.array(hist) / (h * w) 

def get_img(image, norm_size=True, norm_exposure=False):
    '''
    Prepare an image for image processing tasks
    '''
    # flatten returns a 2d grayscale array
    img = [(x * 255).round() for x in image] ; img
    img = np.array(img)

    # resizing returns float vals 0:255; convert to ints for downstream tasks
    if norm_size:
        img = resize(img, (height, width), anti_aliasing=True, preserve_range=True)
    if norm_exposure:
        img = normalize_exposure(img)
    return img

def normalize_exposure(img):
    '''
    Normalize the exposure of an image.
    '''
    img = img
    hist = get_histogram(img)
    # get the sum of vals accumulated by each position in hist
    cdf = np.array([sum(hist[:i+1]) for i in range(len(hist))])
    # determine the normalization values for each unit of the cdf
    sk = np.uint8(255 * cdf)
    # normalize each position in the output image
    height, width = img.shape
    normalized = np.zeros_like(img)
    for i in range(0, height):
        for j in range(0, width):
            normalized[i, j] = sk[int(img[i, j])]
    return normalized.astype(int)

def sift_sim(img1, img2):
    '''
    Use SIFT features to measure image similarity
    '''
    # initialize the sift feature detector
    orb = cv2.ORB_create()

    # get the images
    img_a = img1.astype * 255
    img_b = img2.astype * 255

    print(img_a.shape)
    # img_a = np.resize(img1,(28,28,3))
    # print(img_a.shape)
    print(img_a[0][0])
    plt.imshow(img_a)
    plt.show()
    # find the keypoints and descriptors with SIFT
    kp_a, desc_a = orb.detectAndCompute(img_a, None)
    kp_b, desc_b = orb.detectAndCompute(img_b, None)

    # initialize the bruteforce matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # match.distance is a float between {0:100} - lower means more similar
    matches = bf.match(desc_a, desc_b)
    similar_regions = [i for i in matches if i.distance < 70]
    if len(matches) == 0:
        return 0
    return len(similar_regions) / len(matches)

def jc(img1, img2):
    return jaccard_score(img1.reshape((-1)), img2.reshape((-1)), average='macro')

def cs(img1, img2):
    return 1 - distance.cosine(img1.reshape((-1)), img2.reshape((-1)))

def cw_ssim(img1, img2):
    return SSIM(ImageOps.grayscale(Image.fromarray(img1))).cw_ssim_value(ImageOps.grayscale(Image.fromarray(img2)))

def getDistance(img1, img2, laten_img, typeOfMethod): # return the distance from both pivots
    
    img_01 = [(x * 255).round() for x in img1] ; img_01
    img_01 = np.array(img_01,dtype=np.int32)
    img_02 = [(x * 255).round() for x in img2] ; img_02
    img_02 = np.array(img_02,dtype=np.int32)
    img_latent = [(x * 255).round() for x in laten_img] ; img_latent
    img_latent = np.array(img_latent,dtype=np.int32)

    img1 = np.reshape(img_02, (28,28))
    img2 = np.reshape(img_01, (28,28))
    laten_img = np.reshape(img_latent, (28,28))

    if(typeOfMethod == 'mse'):
        first_distance = mse(img1, laten_img)
        second_distance = mse(img2, laten_img)
    elif(typeOfMethod == 'rmse'):
        first_distance = rmse(img1, laten_img)
        second_distance = rmse(img2, laten_img)
    elif(typeOfMethod == 'psnr'):
        first_distance = psnr(img1, laten_img)
        second_distance = psnr(img2, laten_img)
    elif(typeOfMethod == 'rmse_sw'):
        first_distance, _ = rmse_sw(img1, laten_img)
        second_distance, _ = rmse_sw(img2, laten_img)
    elif(typeOfMethod == 'uqi'):
        first_distance = uqi(img1, laten_img)
        second_distance = uqi(img2, laten_img)
    elif(typeOfMethod == 'ssim'):
        first_distance, _ = ssim(img1, laten_img)
        second_distance, _ = ssim(img2, laten_img)
    elif(typeOfMethod == 'scc'):
        first_distance = scc(img1, laten_img)
        second_distance = scc(img2, laten_img)   
    elif(typeOfMethod == 'rase'):
        first_distance = rase(img1, laten_img)
        second_distance = rase(img2, laten_img)
    elif(typeOfMethod == 'sam'):
        first_distance = sam(img1, laten_img)
        second_distance = sam(img2, laten_img)
    elif(typeOfMethod == 'msssim'):
        first_distance = msssim(img1, laten_img)
        second_distance = msssim(img2, laten_img)
    elif(typeOfMethod == 'vifp'):
        first_distance = vifp(img1, laten_img)
        second_distance = vifp(img2, laten_img)
    elif(typeOfMethod == 'psnrb'):
        first_distance = psnrb(img1, laten_img)
        second_distance = psnrb(img2, laten_img)
    elif(typeOfMethod == 'jc'):
        first_distance = jc(img1, laten_img)
        second_distance = jc(img2, laten_img)
    elif(typeOfMethod == 'cs'):
        first_distance = cs(img1, laten_img)
        second_distance = cs(img2, laten_img)
    elif(typeOfMethod == 'cw-ssim'):
        first_distance = cw_ssim(img1, laten_img)
        second_distance = cw_ssim(img2, laten_img)

    return abs(first_distance - second_distance) , first_distance, second_distance

def collect_data(path):
    path_dir = os.listdir(path)
    acc_images = []
    acc_labels = []
    for font in path_dir:
        x,y,_ = load_data_from_file(path+'/'+font)
        acc_images.append(x)
        acc_labels.append(y)
    return np.array(acc_images).reshape((-1,28,28)), np.array(acc_labels).reshape((-1))

def get_alphabet_pos(alp, font1, font2, labels):
    head_set = []
    tail_set = []
    cur_alphabet = alp
    cur_count_head = 0
    cur_count_tail = 0
    for idx in range(7800): # length of each font
        first_pivot = idx + (first_font*7800)
        second_pivot = idx + (second_font*7800)
        if (cur_count_head == 10 and cur_count_tail == 10): # we want 10 samples
            break
        if (labels[first_pivot] == cur_alphabet and cur_count_head < 10):
            head_set.append(first_pivot)
            cur_count_head += 1
        if (labels[second_pivot] == cur_alphabet and cur_count_tail < 10):
            tail_set.append(second_pivot)
            cur_count_tail += 1
    return head_set, tail_set


# path = 'mini_font/MINI_PAIR_Fonts_complete-in-him+caylee.npz'
font_folder = 'MY_TOTAL_CVAE_z100'
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
        encoder = tf.keras.models.load_model(font_folder+'/encoder_model_z100.h5')
        decoder = tf.keras.models.load_model(font_folder+'/decoder_model_z100.h5')


        path = 'TOTAL_fonts'
        # X_train, y_train, font_label = load_data_from_file(path)  # Total: 15600 (7800 x 2) 7800 each font with 300 each alphabet
        X_train, y_train = collect_data(path)
        X_train = X_train / 255.0
        print(f'X_train: {X_train.shape}, y_train: {y_train.shape}')
        X_train = X_train.reshape((len(X_train), 784))
        labels = y_train
        y_train = to_categorical(y_train)
        n_y = y_train.shape[1]


        z_train = encoder.predict([X_train, y_train]) # Get all the encoded codes


        matrix = []
        synthesized_letters = []
        synthesized_letters_index = []
        syn_label = []

        #                 0     1       2       3       4     5      6     7      8      9       10     11     12     13
        # typeOfMethod = ['mse','rmse','psnr','rmse_sw','uqi','ssim','scc','rase','sam','msssim','vifp','psnrb','cs','cw-ssim']
        typeOfMethod = ['mse','uqi','ssim','vifp','cw-ssim']
        methodFolder = ['MSE','UQI','SSIM','VIFP','CW-SSIM']
        ind = method_number

        first_font = 0
        second_font = 15

        img_it = 0

        saved_first_image = []
        saved_first_label = []
        saved_syn_image = []
        saved_syn_label = []
        saved_second_image = []
        saved_second_label = []
        for i in range(1,27): # 'A'-'Z' or 1-26
            history = []
            temp = []
            # first_pivot = ((first_font*7800) + i) - 1
            # second_pivot = ((second_font*7800) + i ) - 1
            latents_letters = []
            character_label = i#labels[i-1]
            head_set, tail_set = get_alphabet_pos(i, first_font, second_font, labels)
            for a in range(len(head_set)):
                for j in range(0, sides):
                    p = step * j
                    latent = latent_walk(z_train,head_set[a],tail_set[a],p) # for 100 vector font style
                    want_z = np.array(latent)
                    vec = construct_numvec(character_label,want_z)
                    decoded = decoder.predict(vec)
                    # if (method_number == 0):
                    #     plt.subplot(26, sides, 1 + img_it)
                    #     plt.axis('off')
                    img_it +=1
                    latent_image = decoded.reshape(28, 28)
                    actual_first = X_train[head_set[a]].reshape((28,28))
                    actual_second = X_train[tail_set[a]].reshape((28,28))
                    d0, first, second = getDistance(actual_first, actual_second, latent_image, typeOfMethod[ind])
                    # d1, _, _ = getDistance(actual_first, actual_second, latent_image, typeOfMethod[1])
                    # d2, _, _ = getDistance(actual_first, actual_second, latent_image, typeOfMethod[2])
                    # d3, _, _ = getDistance(actual_first, actual_second, latent_image, typeOfMethod[3])
                    # d4, _, _ = getDistance(actual_first, actual_second, latent_image, typeOfMethod[4])
                    # d5, _, _ = getDistance(actual_first, actual_second, latent_image, typeOfMethod[5])
                    # d6, _, _ = getDistance(actual_first, actual_second, latent_image, typeOfMethod[6])
                    # d7, _, _ = getDistance(actual_first, actual_second, latent_image, typeOfMethod[7])
                    # d8, _, _ = getDistance(actual_first, actual_second, latent_image, typeOfMethod[8])
                    # d9, _, _ = getDistance(actual_first, actual_second, latent_image, typeOfMethod[9])
                    # d10, _, _ = getDistance(actual_first, actual_second, latent_image, typeOfMethod[10])
                    # d11, _, _ = getDistance(actual_first, actual_second, latent_image, typeOfMethod[11])
                    # distanceFromPivots2, first1, second2 = getDistance(actual_first, actual_second, latent_image, 'ssim')
                    # plt.imshow(latent_image,cmap='gray')
                    # plt.show()
                    temp.append(d0)
                    latents_letters.append(latent_image)
                    # plt.imshow(latent_image, cmap = 'gray')
                    # print(f'cs: {d0}, f: {first}, s: {second}')
                    # history.append({typeOfMethod[ind]:d0,'first':first,'second':second})
                    history.append({typeOfMethod[ind]:d0})
                    # history.append({
                    #     # typeOfMethod[0]:distanceFromPivots,
                    #     typeOfMethod[1]:d1,
                    #     typeOfMethod[2]:d2,
                    #     typeOfMethod[3]:d3,
                    #     typeOfMethod[4]:d4,
                    #     typeOfMethod[5]:d5,
                    #     typeOfMethod[6]:d6,
                    #     typeOfMethod[7]:d7,
                    #     typeOfMethod[8]:d8,
                    #     typeOfMethod[9]:d9,
                    #     typeOfMethod[10]:d10,
                    #     typeOfMethod[11]:d11
                    # })
                                
                    
                # _plot_loss(history,chr(64+i))

                best_synthesized_letter_index = np.argmin(temp)
                best_index_removed = np.delete(temp, np.argmin(temp),0)
                second_best_index = np.argmin(best_index_removed)
                # print(best_synthesized_letter_index)
                # print(second_best_index)
                # _plot_loss(history,chr(64+i))
                synthesized_letters_index.append(best_synthesized_letter_index)
                synthesized_letters.append(latents_letters)
                syn_label.append(character_label)

                saved_first_image.append(latents_letters[0])
                saved_syn_image.append(latents_letters[best_synthesized_letter_index])
                saved_second_image.append(latents_letters[len(latents_letters)-1])
                saved_first_label.append(character_label)
                saved_syn_label.append(character_label)
                saved_second_label.append(character_label)


        # Create directory
        if (step == 0.5):
            pair_font = font_folder+ '/' + str(first_font+1)+'+'+str(second_font+1)
            dirName = pair_font+'/Synthesized_Letters_z'+str(n_z)+'_'+typeOfMethod[ind]+'_'+str(first_font+1)+'+'+str(second_font+1)+'_L'+str(step)
        else:
            pair_font = font_folder + '/' +str(first_font+1)+'+'+str(second_font+1) + '/'
            dirName = pair_font+'/'+methodFolder[method_number]+'/Synthesized_Letters_z'+str(n_z)+'_'+typeOfMethod[ind]+'_'+str(first_font+1)+'+'+str(second_font+1)+'_L'+str(step)

        try:
            # Create target Directory
            os.mkdir(dirName)
            print("Directory " , dirName ,  " Created ") 
        except FileExistsError:
            print("Directory " , dirName ,  " already exists")

        # Save Latent Space Walk

        synthesized_letters = np.array(synthesized_letters)
        syn_label = np.array(syn_label)
        # print(f'synthesized_letters: {synthesized_letters.shape}')
        # print(f'synthesized_letters_index: {len(synthesized_letters_index)}')


        saved_first_image = np.array(saved_first_image)
        saved_first_label = np.array(saved_first_label)
        saved_syn_image = np.array(saved_syn_image)
        saved_syn_label = np.array(saved_syn_label)
        saved_second_image = np.array(saved_second_image)
        saved_second_label = np.array(saved_second_label)
        npz_dir = dirName+'/'+'z'+str(n_z)+'_'+typeOfMethod[ind]+'_'+str(first_font+1)+'+'+str(second_font+1)+'_L'+str(step)
        np.savez(npz_dir+'_FIRST', saved_first_image, saved_first_label)
        np.savez(npz_dir+'_SYN', saved_syn_image, saved_syn_label)
        np.savez(npz_dir+'_SECOND', saved_second_image, saved_second_label)
        print('SAVED RESULTS!!')
        # print(f'saved_first_image: {saved_first_image.shape}, saved_first_label: {saved_first_label.shape}')
        # print(f'saved_syn_image: {saved_syn_image.shape}, saved_syn_label: {saved_syn_label.shape}')
        # print(f'saved_second_image: {saved_second_image.shape}, saved_second_label: {saved_second_label.shape}')
plt.imshow(np.ones((28,28)))
plt.show()
'''

        # final_syn = []
        saved_first_image = []
        saved_first_label = []
        saved_syn_image = []
        saved_syn_label = []
        saved_second_image = []
        saved_second_label = []
        cur_label = 0
        for k in range(synthesized_letters.shape[0]): # 'A' - 'Z'
            for e in range(0,30,3): # each alphabet has (sides x 10) characters
                for i in range(1,4):
                    if (i == 1):
                        temp_syn = synthesized_letters[k][e]
                        saved_first_image.append(synthesized_letters[k][e]) # save first font
                    elif(i == 2):
                        if(step == 0.5): # only select the 50% 
                            temp_syn = np.concatenate((temp_syn,synthesized_letters[k][e+1]),axis=1)
                            saved_syn_image.append(synthesized_letters[k][e+1]) # save first font

                        else:
                            temp_syn = np.concatenate((temp_syn,synthesized_letters[k][synthesized_letters_index[k]]),axis=1)
                            saved_syn_image.append(synthesized_letters[k][synthesized_letters_index[k]]) # save syn font
                        
                    else:
                        temp_syn = np.concatenate((temp_syn,synthesized_letters[k][e+2]),axis=1)
                        saved_second_image.append(synthesized_letters[k][e+2]) # save second font

                    if (k % sides == 0): # every sides will change label
                        cur_label += 1
                    saved_first_label.append(cur_label)
                    saved_syn_label.append(cur_label)
                    saved_second_label.append(cur_label)
                if (k == 0):
                    final_syn = temp_syn
                else:
                    final_syn = np.concatenate((final_syn,temp_syn),axis=0)

            # if(step == 0.5): 
            #     plt.savefig(dirName+'/'+str(chr(k+65))+'_'+'Naive.png')
            # else:
            #     plt.savefig(dirName+'/'+str(chr(k+65))+'_'+str((100*(synthesized_letters_index[k]/(sides-1))))+'.png')

        # plt.show()
        # final_syn = np.array(final_syn)
        # plt.imshow(final_syn*-1.0,cmap='gray')
        # plt.axis('off')
        # plt.savefig(dirName+'/FINAL_SYN_z'+str(n_z)+str(first_font+1)+'+'+str(second_font+1)+'.png')
        # plt.show()

        saved_first_image = np.array(saved_first_image)
        saved_first_label = np.array(saved_first_label)
        saved_syn_image = np.array(saved_syn_image)
        saved_syn_label = np.array(saved_syn_label)
        saved_second_image = np.array(saved_second_image)
        saved_second_label = np.array(saved_second_label)
        print(f'saved_first_image: {saved_first_image.shape}, saved_first_label: {saved_first_label.shape}')
        print(f'saved_syn_image: {saved_syn_image.shape}, saved_syn_label: {saved_syn_label.shape}')
        print(f'saved_second_image: {saved_second_image.shape}, saved_second_label: {saved_second_label.shape}')

        plt.imshow(saved_first_image[0])
        plt.show()
        # np.savez(dirName+'/'+'z'+str(n_z)+'_'+typeOfMethod[ind]+'_'+str(first_font+1)+'+'+str(second_font+1)+'_L'+str(step), saved_syn_image, saved_syn_label)
        # print('SAVED RESULTS!!')
        
'''