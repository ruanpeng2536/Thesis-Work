import numpy as np
from tensorflow import keras
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, BatchNormalization
from matplotlib import pyplot as plt
from numpy.random import seed
from keras.utils import to_categorical
seed(1)

# Image params
img_rows, img_cols = 28, 28

# Data params
letter_file = "emnist/emnist-letters-train.csv"
test_file = "emnist/emnist-letters-test.csv"
num_classes = 27
# classes = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'
classes = 'ABCDEFGHIJKLMNOPQRSTUVWXYZz'

## Prepare input data
def prep_data(raw):
    y = raw[:, 0]
    out_y = keras.utils.to_categorical(y, num_classes)

    x = raw[:, 1:]
    num_images = raw.shape[0]
    out_x = x.reshape(num_images, img_rows, img_cols, 1)
    out_x = out_x / 255
    return out_x, out_y

## Convert One-Hot-Encoded values back to real values
def decode_label(binary_encoded_label):
    return np.argmax(binary_encoded_label)-1

## Plot an image with it's correct value
def show_img(img,label):
    # img_flip = np.transpose(img, axes=[1,0])
    img_flip = img
    plt.title('Label: ' + str(classes[decode_label(label)]))
    plt.imshow(img_flip, cmap='Greys_r')

    ## Evaluate model with the test dataset
def eval_model(model,test_x,test_y):
    result = model.evaluate(test_x, test_y)
    print("The accuracy of the model is: ",result[1])
    return result

## Plot the training history
def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'ro', label='Validation loss')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


# Create a more complex model
# As the architectura decision for the project
def create_complex_model(input_size,output_size):
    model = Sequential()

    model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (input_size[0], input_size[1], input_size[2])))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size = 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(64, kernel_size = 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size = 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, kernel_size = 4, activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(output_size, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
    
    return model

def load_data_from_file(file_dir):
    # returns: train_images , train_labels 
    data = np.load(file_dir)
    return data['arr_0'], data['arr_1']

# ===================== TRAIN ========================
# letter_data = np.loadtxt(letter_file, skiprows=1, delimiter=',')
# x, y = prep_data(letter_data)
# print(x.shape)
# print(y.shape)

print('### my_Fonts ###')
path = './my_fonts'
print(os.listdir(path)[0])
x_train_test, y_train_test = load_data_from_file(path+'/'+os.listdir(path)[0])
# allFonts = x_train_test.reshape(*x_train_test.shape[:1], -1) / 255
allFonts = x_train_test
# for i in range(30):
#     plt.imshow(x_train_test[i])
#     plt.show()
#     print(chr(y_train_test[i]+64))

x = []
y = []
test_x = []
test_y = []
number_of_test_image = 77910
for idx in range(allFonts.shape[0]):
    if (idx < number_of_test_image):
        test_x.append(x_train_test[idx])
        test_y.append(y_train_test[idx])
    else:
        x.append(x_train_test[idx])
        y.append(y_train_test[idx])

x = np.asarray(x, dtype=np.float32) / 255
x = np.reshape(x, (-1, img_rows, img_cols, 1))
y = np.asarray(y, dtype=np.float32)
y = keras.utils.to_categorical(y, num_classes)

test_x = np.asarray(test_x, dtype=np.float32) / 255
test_x = np.reshape(test_x, (-1, img_rows, img_cols, 1))
test_y = np.asarray(test_y, dtype=np.float32)
test_y = keras.utils.to_categorical(test_y, num_classes)


print(x.shape)
print(y.shape)
print(test_x.shape)
print(test_y.shape)


# for i in range(30):
#     plt.imshow(x[i])
#     plt.show()
#     print(chr(y[i]+64))

# fig = plt.figure(figsize=(17,4.5))
# for idx in range(30, 60):
#    fig.add_subplot(3,10,(idx-30)+1)
#    plt.axis('off')
#    show_img(np.squeeze(x[idx]),y[idx])
# plt.subplots_adjust(wspace=0.3, hspace=0.3)
# plt.show()

# ===================== TEST ========================
# test_data = np.loadtxt(test_file, skiprows=1, delimiter=',')
# test_x, test_y = prep_data(test_data)
# print(test_x.shape)
# print(test_y.shape)



batch_size = 64

complex_model = create_complex_model([img_rows, img_cols,1],len(classes))
complex_history = complex_model.fit(x, y,
          batch_size = batch_size,
          epochs = 15,
          validation_split = 0.1)

complex_model.save("myAllFonts_model.h5")
plot_history(complex_history)
eval_model(complex_model,test_x,test_y)
