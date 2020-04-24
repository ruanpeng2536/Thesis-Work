# import os
# os.environ["PATH"] += os.pathsep + f'C:\Program Files (x86)\Graphviz2.38\bin'
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
import tensorflow as tf

from mlxtend.plotting import plot_confusion_matrix

import tensorflow as tf
import imageio
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.utils import to_categorical
from tensorflow import keras
import os
from keras.preprocessing.image import load_img
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score

classes = 'ABCDEFGHIJKLMNOPQRSTUVWXYZz'
# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('myAllFonts_model.h5')
num_classes = 27
img_rows, img_cols = 28, 28
# Show the model architecture
# new_model.summary()

def load_data_from_file(file_dir):
    # returns: train_images , train_labels 
    data = np.load(file_dir)
    return data['arr_0'], data['arr_1']

def load_data_result(file_dir):
    # returns: train_images , train_labels 
    data = np.load(file_dir)
    return data['arr_0']
def decode_label(binary_encoded_label):
    return np.argmax(binary_encoded_label)-1

def calculate_CFM(c):
    all_CF_results = []
    n = c.shape[0]
    for i in range(n): # 0 - 25
        tp = c[i][i]
        sumCol = 0
        sumRow = 0
        sumRowCol = 0
        for l in range(n):
            sumCol += c[l][i]
        for l in range(n):
            sumRow += c[i][l]
        for l in range(n):
            for k in range(n):
                sumRowCol += c[l][k]
        fp = sumCol - tp
        fn = sumRow - tp
        tn = sumRowCol - tp - fp - fn
        all_CF_results.append((tp, fp, fn, tn))
    return all_CF_results


def sumArray(arr,pos):
    s = 0
    for i in range(len(arr)):
        s += arr[i][pos]
    return s
# print('### my_Fonts ###')
# path = './my_fonts'
# print(os.listdir(path)[0])
# x_train_test, y_train_test = load_data_from_file(path+'/'+os.listdir(path)[0])



path = 'Numpy_results'
# path = 'gen_data/nonCVAE(RESULT).npz'
confusion_matrix = []
for n in range(12):
    # n = 3


    x_train_test = load_data_result(path + '/' + os.listdir(path)[n])
    # x_train_test = load_data_result(path)
    # print('x_train_test: ', x_train_test[0])
    allFonts = x_train_test / 255.0
    # print(os.listdir(path)[n])
    

    all_results = np.reshape(allFonts, (-1, img_rows, img_cols, 1))
    result = new_model.predict(all_results) # get all the results in one go!
    # for i in range(result.shape[0]):
    #     print('result: ',result[i])
    #     plt.imshow(allFonts[i])
    #     plt.show()

    
    classes_result = '_ABCDEFGHIJKLMNOPQRSTUVWXYZ' # 1 - 26
    classes_real = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' # 0 - 25
    np.zeros((2,4))
    letters_correct = np.zeros((26,26),dtype=np.uint8) # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    correct = 0

    ground_truth_label = list(range(0,26))
    counter = 0

    y_true = []
    y_pred = []
    for i in range(10):
    # for i in range(10):
        for k in range(26):
            cur_idx = counter%26 + 1
            y_true.append(cur_idx)
            # for r in range(4): # repeat 4 times (ONLY for CVAE-by-font results)
            step = i * 26 + k
            result_index = np.where(result[step] == np.amax(result[step]))
            y_pred.append(result_index[0][0])
            # print(f'cur_idx: {cur_idx}, result_index: {result_index[0][0]}, counter: {counter%26}')
            # plt.imshow(allFonts[step])
            # plt.show()
            if (cur_idx == result_index[0][0]): # check for overall correctness
                correct += 1
            # letters_correct[result_index[0][0]-1] += 1
            else:
                print(f'Actual: {classes_result[cur_idx]}, Predict: {classes_result[result_index[0][0]]}')

                # print(f'Predict: {classes_result[result_index[0][0]]}')
                #letters_correct[result_index[0][0]-1] += 1

            # print(f'cur_idx: {cur_idx}, result_index: {result_index[0][0]}')
            letters_correct[cur_idx-1][result_index[0][0]-1] += 1
            counter += 1

    # for k in range(len(letters_correct)):
    #     print(f'{classes_real[k]}: {(letters_correct[k]/10)*100} % ')
    print(f'Total acc: {(correct/(result.shape[0]))*100} % ')
    confusion_matrix.append(letters_correct)

    multiclass = np.array(letters_correct)
    allCF_scores = calculate_CFM(multiclass)

    tp = sumArray(allCF_scores,0) 
    fp = sumArray(allCF_scores,1) 
    fn = sumArray(allCF_scores,2) 
    tn = sumArray(allCF_scores,3) 

    # print(f'tp: {tp}, fp: {fp}, fn: {fn}, tn: {tn}')
    recall = tp / (tp +fn)
    precision = tp / (tp + fp)
    f_measure = (2*recall*precision) / (recall+precision)
    accuracy = (tp + tn) / (tp + fn + fp + tn)
    # print(f'Recal: {recall}, Precision: {precision}, F-measure: {f_measure}, Accuracy: {accuracy}')

    
    # print('multiclass: ',multiclass.shape)
    recallScore = recall_score(y_true, y_pred, average='macro') 
    print('[R]ecallScore: ', recallScore)
    precisionScore = precision_score(y_true, y_pred, average='macro') 
    print('[P]recisionScore: ', precisionScore)
    f1Score = f1_score(y_true, y_pred, average='macro') 
    print('[F]1Score: ', f1Score)
    acc = accuracy_score(y_true, y_pred) 
    print('Acc: ', acc)
    


    # print (calculate_CFM(multiclass))
    class_names = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q', 'R','S','T','U','V','W','X','Y','Z']
    fig, ax = plot_confusion_matrix(conf_mat=multiclass,
                                    colorbar=True,
                                    class_names=class_names)

                                    # show_absolute=True,
                                    # show_normed=False,


    # ax.set_title(os.listdir(path)[n].split('.')[0], fontsize=12, fontweight='bold')
    ax.set_title(os.listdir(path)[n], fontsize=12, fontweight='bold')
    ax.set_xlabel('Output Class')
    ax.set_ylabel('Target Class')
    plt.show()
    # plt.imsave(os.listdir(path)[n]+'.png', fig)



#=============================================

# show_test_x = []
# test_x = []
# test_y = []
# number_of_test_image = 10
# for idx in range(allFonts.shape[0]):
#     if (idx < number_of_test_image):
#         show_test_x.append(x_train_test[idx])
#         test_x.append(x_train_test[idx])
#         test_y.append(y_train_test[idx])
#     else:
#         break


# show_test_x = np.asarray(show_test_x, dtype=np.float32) / 255
# test_x = np.asarray(test_x, dtype=np.float32) / 255
# test_x = np.reshape(test_x, (-1, img_rows, img_cols, 1))
# print(test_x.shape)
# test_y = np.asarray(test_y, dtype=np.float32)
# test_y = keras.utils.to_categorical(test_y, num_classes)
# classes_result = '_ABCDEFGHIJKLMNOPQRSTUVWXYZ' # 1 - 26
# classes_real = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ_' # 0 - 25

# result = new_model.predict(test_x)

# # print(test_x[0])
# # print(test_x.shape)
# s = np.reshape(test_x[0], (img_rows, img_cols))
# print(s.shape)
# plt.imshow(s)
# plt.show()

# for im_path in glob.glob("gen_data/*.png"):
#     image = load_img(im_path)
#     letter = np.asarray(image, dtype=np.float32)
#     # alphabet = (letter * -(1/255.0)) + 1
#     print(letter.shape)
#     plt.imshow(letter)
#     plt.show()



# for i in range(number_of_test_image):
#     print(f"=======  =======")
#     plt.imshow(show_test_x[i])
#     plt.show()
#     print(decode_label(test_y[i]))
#     print('real test_y(letter): ', classes_real[decode_label(test_y[i])])
#     print('real test_y(index): ', decode_label(test_y[i]))
#     # top3_result = result[0].argsort()[-3:][::-1]
#     # print(top3_result[0])
#     # print(top3_result[1])
#     result_index = np.where(result[i] == np.amax(result[i]))
#     print('result(letter): ', classes_result[result_index[0][0]])
#     print('result(index): ', result_index[0][0])

'''
# Predict
o = 1
letters = 'aABCDEFGHIJKLMNOPQRSTUVWXYZ';
correct = 0
# for im_path in glob.glob("gen_data/All Generated Fons/agatha+a-yummy-apology/Generate_letters/A-Z4/*.png"): # agatha+a-yummy-apology/A-Z
for im_path in glob.glob("gen_data/*.png"):
    img = cv2.imread(im_path)
    letter = np.asarray(img, dtype=np.float32)
    alphabet = (letter * -(1/255.0)) + 1
    # plt.imshow(alphabet)
    # plt.show()
    # alphabet = (letter * (1/255.0))
    alp = np.resize(alphabet,(28,28,1))
    sample = [alp]
    s = np.asarray(sample)
    result = new_model.predict(s)
    
    first_result = result[0].argsort()[-3:][::-1]
    # print(first_result)
    # print(first_result)
    # if (o == 0):
    #     o += 1
        # plt.imshow(alphabet)
        # plt.show()
        # print(alp)
        # print(alp.shape)

    plt.imshow(alphabet)
    plt.show()
    print(f"======= {letters[o]} =======")
    print(letters[first_result[0]])
    print(f'(1)letter: {letters[first_result[0]]}, at: {result[0][first_result[0]]*100}%')
    print(f'(2)letter: {letters[first_result[1]]}, at: {result[0][first_result[1]]*100}%')
    print(f'(3)letter: {letters[first_result[2]]}, at: {result[0][first_result[2]]*100}%')
    
    if (letters[o] == letters[first_result[0]] or letters[o] == letters[first_result[1]] or letters[o] == letters[first_result[2]]):
        correct += 1
        print("$$$ YEAH!! $$$")
    o += 1
        
    # do whatever with the image here

print ('correct: ', correct)
# result = new_model.predict()
'''