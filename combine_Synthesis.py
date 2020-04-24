import os,glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import imutils
import itertools
from itertools import permutations
from itertools import combinations
from PIL import ImageDraw, ImageFont
from PIL import Image as I
import random
import matplotlib

def get_img_object(img_dir):
    img = cv2.imread(img_dir,cv2.IMREAD_UNCHANGED)
    b,g,r,a = cv2.split(img)
    return a

def save_image(img_dir, des, op, img):
    rename_image = image_dir.split('/')
    folder = des + '/'
    cv2.imwrite(folder + rename_image[len(rename_image)-1].split('.')[0]+'_'+op+'.png', img)


# @@@@@@@@@@@@@@@@@@@@@@@@@ Group 1 operations @@@@@@@@@@@@@@@@@@@@@@@@@

def morphology(img, m, n, op):
    kernel = np.ones((m,n),np.uint8)
    if (op == 'erosion'): # 0 - 3
        return cv2.erode(img,kernel,iterations = 1)
    elif (op == 'dilation'): # 0 - 3
        return cv2.dilate(img,kernel,iterations = 1)
    elif (op == 'opening'): # 1 - 3
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    elif (op == 'closing'): # 1 - 3
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    elif (op == 'gradient'): # 1 - 3 (NO 1,1)
        return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

def hit_and_miss(img, kernels, kernel_number): # 2 binary kernel (10,136)
    n= kernel_combination[kernel_number]
    t1,t2,t3 = int(n[0]),int(n[1]),int(n[2])
    t4,t5,t6 = int(n[3]),int(n[4]),int(n[5])
    t7,t8,t9 = int(n[6]),int(n[7]),int(n[8])
    styled_kernel = np.array([[t1,t2,t3],
                              [t4,t5,t6],
                              [t7,t8,t9]])
    return cv2.morphologyEx(img, cv2.MORPH_HITMISS, styled_kernel)


# @@@@@@@@@@@@@@@@@@@@@@@@@ Group 2 operations @@@@@@@@@@@@@@@@@@@@@@@@@

def shearing(img,isShearingLeft):
    '''
    shear left range [0.6,0.9] 
    shear right range [0.1,0.4]
    '''
    y, x = img.shape[:2]
    distance_from_origin = 0.6

    s_l = [0,0];  s_r = [y-1,0]      # source top-left corner; source top-right corner
    s_b = [0,x-1]                    # source bottom-left corner

    if(isShearingLeft):
        d_l = [0,0];  d_r = [int(distance_from_origin*(x-1)),0]   # destination top-left corner ; destination top-right corner
        d_b = [int((1-distance_from_origin)*(x-1)),y-1]           # destination bottom-left corner
    else:
        d_l = [int(distance_from_origin*(x-1)),0];  d_r = [x-1,0]   # destination top-left corner ; destination top-right corner
        d_b = [0,y-1]                                               # destination bottom-left corner


    src_points = np.float32([s_l, s_r, s_b])
    dst_points = np.float32([d_l, d_r, d_b])
    affine_matrix = cv2.getAffineTransform(src_points, dst_points)
    img_output = cv2.warpAffine(img, affine_matrix, (y,x))
    return img_output

# Zoom in/out with shape (28x28)
def Zoom(img, level):
    zoom_img = cv2.resize(img,None,fx=level,fy=level)
    x,y = zoom_img.shape
    image_size = 28
    if (level > 1):  # 1.1 - 1.4
        x1 = int((x-image_size)*0.5)
        y1 = int((y-image_size)*0.5)
        x2 = image_size+x1
        y2 = image_size+y1
        return zoom_img[y1:y2, x1:x2]
    else: # 0.6 - 0.9
        temp = np.zeros((image_size,image_size),dtype='int')
        x1 = int((image_size-x)*0.5)
        y1 = int((image_size-y)*0.5)
        x2 = x+x1
        y2 = y1
        x3 = x1
        y3 = y+y1
        x4 = x2
        y4 = y3

        row = 0
        col = 0
        test = np.zeros((x,y),dtype='int')
        for i in range(len(temp)):
            if (y1 <= i <= y3):
                for k in range(len(temp[i])):
                    if(x1 <= k < x2):
                        temp[i][k] = zoom_img[row%x][col%y]
                        col += 1
                row += 1
        return temp

def dictionary(command):
    dictionary = {
        1:'erosion',
        2:'dilation',
        3:'opening',
        4:'closing',
        5:'gradient',
        6:'hit and miss',
        7:'shaering',
        8:'zoom',
    }
    return dictionary[command]

def dictionary_operaion(command):
    dictionary = {
        1:'-E',
        2:'-D',
        3:'-S',
        4:'-Z',
        5:'-G'
    }
    return dictionary[command]



# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ WORKING AREA $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

class Image():
    def __init__(self, images):
        self.images = images
        self.operation = '_'

    def set_operation(self, operation):
        self.operation = operation

    def get_operation(self):
        return self.operation

    def get_image(self):
        return self.images

class Erosion(Image):
    def __init__(self, images):
        super().__init__(images)
        self.isOriginal = False
        if isinstance(images, list): # in case images passed from other operation
            self.images = images
        else: # in case images is original
            self.isOriginal = True
            self.images = images # because dilation has 4 posibilities
        self.operation = 'e'
        self.generate_Erosion()

    def generate_Erosion(self):
        combination = []
        count = 0
        isZero = False

        if (self.isOriginal):
            for i in range(2):
                for k in range(2):
                    if (not isZero):
                        isZero = True
                        continue
                        isZero = True
                    elif(i != 0 and k != 0): # (0,0) and (3,2) not included
                        kernel = np.ones((i,k),np.uint8)
                        image_name = self.images.get_operation() + self.operation
                        gen_img = Image(cv2.erode(self.images.get_image(),kernel,iterations = 1))
                        gen_img.set_operation(image_name+str(i)+str(k))
                        combination.append(gen_img)
        else:
            for i in range(2):
                for k in range(2):
                    for letter in self.images:
                        if (not isZero):
                            isZero = True
                            continue
                            isZero = True
                        elif(i != 0 and k != 0): # (0,0) and (3,2) not included
                            kernel = np.ones((i,k),np.uint8)
                            image_name = letter.get_operation() + self.operation
                            gen_img = Image(cv2.erode(letter.get_image(),kernel,iterations = 1))
                            gen_img.set_operation(image_name+str(i)+str(k))
                            combination.append(gen_img)

        self.images = combination

class Dilation(Image):
    def __init__(self, images):
        super().__init__(images)
        self.isOriginal = False
        if isinstance(images, list): # in case images passed from other operation
            self.images = images
        else: # in case images is original
            self.isOriginal = True
            self.images = images # because dilation has 4 posibilities
        self.operation = 'd'
        self.generate_Dilation()

    def generate_Dilation(self):
        combination = []
        isZero = False
        if (self.isOriginal):
            for i in range(4):
                for k in range(3):
                    if (not isZero):
                        isZero = True
                        continue
                        isZero = True
                    elif((i != 0 and k != 0) and not(i == 3 and k ==2)): # (0,0) and (3,2) not included
                        kernel = np.ones((i,k),np.uint8)
                        image_name = self.images.get_operation() + self.operation
                        gen_img = Image(cv2.dilate(self.images.get_image(),kernel,iterations = 1))
                        gen_img.set_operation(image_name+str(i)+str(k))
                        combination.append(gen_img)
        else:
            for i in range(4):
                for k in range(3):
                    for letter in self.images:
                        if (not isZero):
                            isZero = True
                            continue
                            isZero = True
                        elif((i != 0 and k != 0) and not(i == 3 and k ==2)): # (0,0) and (3,2) not included
                            kernel = np.ones((i,k),np.uint8)
                            image_name = letter.get_operation() + self.operation
                            gen_img = Image(cv2.dilate(letter.get_image(),kernel,iterations = 1))
                            gen_img.set_operation(image_name+str(i)+str(k))
                            combination.append(gen_img)

        self.images = combination

class Zoom(Image):
    def __init__(self, images):
        super().__init__(images)
        self.isOriginal = False
        if isinstance(images, list): # in case images passed from other operation
            self.images = images
        else: # in case images is original
            self.isOriginal = True
            self.images = images # because zoom has 4 posibilities (2 IN and 2 OUT)
        self.operation = 'z'
        self.generate_Zoom()

    def generate_Zoom(self):
        combination = []
        zoom_in_out = [[1.1,1.2],[0.8,0.9]]
        image_size = 28

        if(self.isOriginal):
                # Run every zoom level and apply to all of the input images
            for in_out in range(2):
                for zoom_level in zoom_in_out[in_out]:
                    level = zoom_level
                    zoom_img = cv2.resize(self.images.get_image(),None,fx=level,fy=level)
                    x,y = zoom_img.shape
                    image_name = self.images.get_operation() + self.operation
                    if (level > 1):  # 1.1 - 1.4
                        x1 = int((x-image_size)*0.5)
                        y1 = int((y-image_size)*0.5)
                        x2 = image_size+x1
                        y2 = image_size+y1
                        gen_img = Image(zoom_img[y1:y2, x1:x2])
                        gen_img.set_operation(image_name+str(int(zoom_level*10)))
                        combination.append(gen_img)

                    else: # 0.6 - 0.9
                        temp = np.zeros((image_size,image_size),dtype='int')
                        x1 = int((image_size-x)*0.5)
                        y1 = int((image_size-y)*0.5)
                        x2 = x+x1
                        y2 = y1
                        x3 = x1
                        y3 = y+y1
                        x4 = x2
                        y4 = y3

                        row = 0
                        col = 0
                        test = np.zeros((x,y),dtype='int')
                        for i in range(len(temp)):
                            if (y1 <= i <= y3):
                                for k in range(len(temp[i])):
                                    if(x1 <= k < x2):
                                        temp[i][k] = zoom_img[row%x][col%y]
                                        col += 1
                                row += 1
                        gen_img = Image(temp)
                        gen_img.set_operation(image_name+str(int(zoom_level*10)))
                        combination.append(gen_img)
        else:
            for in_out in range(2):
                for zoom_level in zoom_in_out[in_out]:
                    for letter in self.images:
                        level = zoom_level
                        zoom_img = cv2.resize(letter.get_image(),None,fx=level,fy=level)
                        x,y = zoom_img.shape
                        image_name = letter.get_operation() + self.operation
                        if (level > 1):  # 1.1 - 1.4
                            x1 = int((x-image_size)*0.5)
                            y1 = int((y-image_size)*0.5)
                            x2 = image_size+x1
                            y2 = image_size+y1
                            gen_img = Image(zoom_img[y1:y2, x1:x2])
                            gen_img.set_operation(image_name+str(int(zoom_level*10)))
                            combination.append(gen_img)

                        else: # 0.6 - 0.9
                            temp = np.zeros((image_size,image_size),dtype='int')
                            x1 = int((image_size-x)*0.5)
                            y1 = int((image_size-y)*0.5)
                            x2 = x+x1
                            y2 = y1
                            x3 = x1
                            y3 = y+y1
                            x4 = x2
                            y4 = y3

                            row = 0
                            col = 0
                            test = np.zeros((x,y),dtype='int')
                            for i in range(len(temp)):
                                if (y1 <= i <= y3):
                                    for k in range(len(temp[i])):
                                        if(x1 <= k < x2):
                                            temp[i][k] = zoom_img[row%x][col%y]
                                            col += 1
                                    row += 1
                            gen_img = Image(temp)
                            gen_img.set_operation(image_name+str(int(zoom_level*10)))
                            combination.append(gen_img)

        self.images = combination # add all generated images to the newly created Image object



class Shearing(Image):
    def __init__(self, images):
        super().__init__(images)
        self.isOriginal = False
        if isinstance(images, list): # in case images passed from other operation
            self.images = images
        else: # in case images is original
            self.isOriginal = True
            self.images = images # because shearing has 8 posibilities (4 LEFT and 4 RIGHT)
        self.operation = 's'
        self.generate_Shearing()

    def generate_Shearing(self):
        '''
        shear left range [0.6,0.9]
        shear right range [0.1,0.4]
        '''
        combination = []
        leftRight = [0,1]
##        left_and_right_shear = [[0.1,0.2,0.3,0.4],[0.6,0.7,0.8,0.9]] # right and then left
        left_and_right_shear = [[0.1,0.2],[0.8,0.9]] # right and then left
        if (self.isOriginal):
            for isShearingLeft in leftRight: #[0,1]
                for angle in left_and_right_shear[isShearingLeft]:
                    img = self.images.get_image()
                    y, x = img.shape[:2]
                    distance_from_origin = angle
                    s_l = [0,0];  s_r = [y-1,0]      # source top-left corner; source top-right corner
                    s_b = [0,x-1]                    # source bottom-left corner
                    image_name = self.images.get_operation() + self.operation

                    if(isShearingLeft):
                        d_l = [0,0];  d_r = [int(distance_from_origin*(x-1)),0]   # destination top-left corner ; destination top-right corner
                        d_b = [int((1-distance_from_origin)*(x-1)),y-1]           # destination bottom-left corner
                    else:
                        d_l = [int(distance_from_origin*(x-1)),0];  d_r = [x-1,0]   # destination top-left corner ; destination top-right corner
                        d_b = [0,y-1]                                               # destination bottom-left corner

                    src_points = np.float32([s_l, s_r, s_b])
                    dst_points = np.float32([d_l, d_r, d_b])
                    affine_matrix = cv2.getAffineTransform(src_points, dst_points)
                    img_output = cv2.warpAffine(img, affine_matrix, (y,x))
                    gen_img = Image(img_output)
                    gen_img.set_operation(image_name+str(int(angle*10)))
                    combination.append(gen_img)
        else:
            for image in self.images: # loop to all the input images
                for isShearingLeft in leftRight: #[0,1]
                    for angle in left_and_right_shear[isShearingLeft]:
                        img = image.get_image()
                        y, x = img.shape[:2]
                        distance_from_origin = angle
                        s_l = [0,0];  s_r = [y-1,0]      # source top-left corner; source top-right corner
                        s_b = [0,x-1]                    # source bottom-left corner
                        image_name = image.get_operation() + self.operation

                        if(isShearingLeft):
                            d_l = [0,0];  d_r = [int(distance_from_origin*(x-1)),0]   # destination top-left corner ; destination top-right corner
                            d_b = [int((1-distance_from_origin)*(x-1)),y-1]           # destination bottom-left corner
                        else:
                            d_l = [int(distance_from_origin*(x-1)),0];  d_r = [x-1,0]   # destination top-left corner ; destination top-right corner
                            d_b = [0,y-1]                                               # destination bottom-left corner

                        src_points = np.float32([s_l, s_r, s_b])
                        dst_points = np.float32([d_l, d_r, d_b])
                        affine_matrix = cv2.getAffineTransform(src_points, dst_points)
                        img_output = cv2.warpAffine(img, affine_matrix, (y,x))
                        gen_img = Image(img_output)
                        gen_img.set_operation(image_name+str(int(angle*10)))
                        combination.append(gen_img)

        print('combination: ',len(combination))
        self.images = combination # add all generated images to the newly created Image object

class GaussianBlur(Image):
    def __init__(self, images):
        super().__init__(images)
        self.isOriginal = False
        if isinstance(images, list): # in case images passed from other operation
            self.images = images
        else: # in case images is original
            self.isOriginal = True
            self.images = images # because Gaussian blur has 2 posibilities (3x3 amd 1x3 kernel)

        #print("Gauss[IN]: ", self.images[0].get_image().dtype)
        self.operation = 'g'
        self.generate_GaussianBlur()

    def generate_GaussianBlur(self):
        combination = []
        if(self.isOriginal):
            img = self.images.get_image().astype(np.uint8)
            for k in range(2):
                image_name = self.images.get_operation() + self.operation
                if (k == 0):
                    gen_img = Image(cv2.GaussianBlur(img,(3,3),0))
                    gen_img.set_operation(image_name+'g')
                else:
                    gen_img = Image(cv2.blur(img,(1,3)))
                    gen_img.set_operation(image_name+'b')
                combination.append(gen_img)
        else:
            for letter in self.images: # apply hit and miss to all the input images
                img = letter.get_image().astype(np.uint8)
                for k in range(2):
                    image_name = letter.get_operation() + self.operation
                    if (k == 0):
                        gen_img = Image(cv2.GaussianBlur(img,(3,3),0))
                        gen_img.set_operation(image_name+'g')
                    else:
                        gen_img = Image(cv2.blur(img,(1,3)))
                        gen_img.set_operation(image_name+'b')
                    combination.append(gen_img)
        self.images = combination # add all generated images to the newly created Image object

def save_images(letter, des, img):
    folder = des + '/'
    for element in img: # save all the images
        cv2.imwrite(folder + letter + element.get_operation() +'.png', element.get_image())

def generate_permutaion_operation():
    dic = {
    '1':2,
    '2':2,
    '3':4,
    '4':2,
    '5':2,
    }

    s = 0

    summ = 0
    stuff = list(range(1,6))
    permutation_operations = []
##    print('stuff: ',stuff)
    for i in range(1,6):
        perm = combinations(stuff,i)#permutations
        c = 0
        s2 = 0
        for i in list(perm):
            c += 1
            permutation_operations.append(i)
            s1 = 1
            for p in i:
                s1 *= dic[str(p)]
            s2 += s1
        summ += s2
        s += c



    '''

    OPERATION (unit in generated image)
    (1)Erosion = 5
    (2)Dilation = 5
    (3)Shearing(left) = 3
       Shearing(right) = 3
    (4)Zoom(in) = 2
       Zoom(out) = 2
    (5)Gaussian Blur = 2
    =>Total operation: 5291

    Total training dataset: 5291 * 26 = 137566

    '''
##    print(s)
    print('summ: ',summ)
    return permutation_operations


def recursive_operation(op, pointer, imgs):
    if(pointer == len(op)):
        return imgs
    elif(op[pointer] == 1):
        return recursive_operation(op, pointer+1, Erosion(imgs).get_image())
    elif(op[pointer] == 2):
        return recursive_operation(op, pointer+1, Dilation(imgs).get_image())
    elif(op[pointer] == 3):
        return recursive_operation(op, pointer+1, Shearing(imgs).get_image())
    elif(op[pointer] == 4):
        return recursive_operation(op, pointer+1, Zoom(imgs).get_image())
    elif(op[pointer] == 5):
        return recursive_operation(op, pointer+1, GaussianBlur(imgs).get_image())

def text_phantom(font_dir, text, size):
    # Availability is platform dependent
    font = font_dir

    # Create font
    pil_font = ImageFont.truetype(font + ".ttf", size=size,
                                  encoding="unic")
    text_width, text_height = pil_font.getsize(text)

    # create a blank canvas with extra space between lines
    canvas = I.new('L', [size, size], (255))

    # draw the text onto the canvas
    draw = ImageDraw.Draw(canvas)
    offset = ((size - text_width) // 2,
              (size - text_height) // 2 -2)
    white = "#000000"
    draw.text(offset, text, font=pil_font, fill=white,align="center")

    # Convert the canvas into an array with values in [0, 255] black background
    return 255-np.asarray(canvas)

def get_alphabets(file_dir):
  alphabets = []
  image_size = 28
  print(file_dir)
#   for i in range(97,123): # 'a' to 'z'
##  font_folder = file_dir.split('\\')[1].split('.')[0]
  font_folder = file_dir.split('.')[0]
  if not os.path.exists(font_folder):
    os.makedirs(font_folder)
  for i in range(65,91): # 'A' to 'Z'
    alp = text_phantom(file_dir.split('.')[0],chr(i), image_size)
    matplotlib.image.imsave(font_folder+'/'+chr(i)+'.png', alp*-1., cmap='gray')
    alphabets.append(alp)
  return np.asarray(alphabets)

def load_data(file_dir):
    # returns: train_images , train_labels
    data = np.load(file_dir)
    print(data['arr_1'].shape)
    return data['arr_0'], data['arr_1'], data['arr_2']

def run():
    import time
    train_images = [] # for images
    train_labels = [] # for labels
    train_style_labels = []
    combined = []
    operations = generate_permutaion_operation()
    font_label = 0
    path = os.listdir("fonts/")

    fontNumber = 1
    for folder in range(12):
        if (folder == fontNumber):
            for im_path in os.listdir('fonts/' + path[folder]):
                alphabets = get_alphabets('fonts/' + path[folder]+'/'+im_path)

                for permute in operations: # each permutation
                   print(permute)
                   for character in range(97,123): # 'a' to 'z'
                       img1 = alphabets[character-97]
                       image_object1 = Image(img1)
                       processed_image1 = recursive_operation(permute, 0, image_object1)

                       for im in processed_image1:
                           combined.append((im.get_image(),(character-97)+1,font_label))


                # for character in range(97,123):
                #     img1 = alphabets[character-97]
                #     image_object1 = Image(img1)
                #     processed_image1 = Shearing(image_object1).get_image()
                #     for im in processed_image1:
                #         # combined.append((im.get_image(),(character-97)+1))
                #         combined.append((im.get_image(),font_label))


            font_label += 1

    # random.shuffle(combined) # shuffle tuples
    print('combined: ', len(combined))
    print(combined[0])
    for i in range(len(combined)):
        train_images.append(combined[i][0])
        train_labels.append(combined[i][1])
        train_style_labels.append(combined[i][2])
    train_images = np.array(train_images,dtype=np.uint8) #as emnist
    train_labels = np.array(train_labels,dtype=np.uint8) #as emnist
    train_style_labels = np.array(train_style_labels,dtype=np.uint8) #as emnist

    np.savez('MINI_PAIR_Fonts' , train_images, train_labels, train_style_labels)
    print('FILES SAVED!!')





def reduce_size(arr,flag):
##    print('arr[PRE]: ',arr.shape)
    fonts = []

    for f in range(2):
        start = int(f*(93496 / 2))
        stop = int((start + 46748)-1)
##        print(f'start: {start}, stop: {stop}')
        font = arr[start : stop]
        for l in range(26):
            start = int(l*1798)
            stop = int((start + 1798)-1)
            letter = font[start : stop]
            fonts.append(letter[0:300])

##    for f in range(2):
##        start = f*93496
##        stop = (start + 93496)-1
##        font = arr[start : stop]
##        for l in range(26):
##            start = l*3596
##            stop = (start + 3596)-1
##            letter = font[start : stop]
##            print('letter: ', len(letter))
##            fonts.append(letter[0:300])
##    print('arr[POST]: ',np.array(fonts).shape)
    if (flag == 0):
        print('X')
        return np.array(fonts).reshape((-1,28,28))
    else:
        return np.array(fonts).reshape((-1))



def test(file_name):
    x,y,z = load_data(file_name+'.npz')
    num = 0
    print('x: ',x.shape)
    print('y: ',y.shape)
    print('z: ',z.shape)

    tx = reduce_size(x, 0)
    ty = reduce_size(y, 1)
    tz = reduce_size(z, 1)
    print('ti: ', tx.shape)
    print('tl: ', ty.shape)
    print('tsl: ', tz.shape)


    np.savez('MINI_PAIR_Fonts_'+ file_name, tx, ty, tz)
    print('FILES SAVED!!')

    for k in tx:
        plt.imshow(k)
        plt.show()

##    for num in range(7799):
##        print('tl[0]: ', ty[num])
##        print('tsl[0]: ', tz[num])
##        plt.imshow(tx[num])
##        plt.show()
##
##    print('tl[7800]: ', ty[7800])
##    print('tsl[7800]: ', tz[7800])
##    plt.imshow(tx[7800])
##    plt.show()

##    count = 0
##    c =0
##    for i in range(x.shape[0]):
##
##        if (z[i] != c):
##            c = z[i]
##            print('count: ', count)
##            count = 0
##            break
##        count += 1
####        print('y: ',y[i])
####        print('z: ',z[i])
##    plt.imshow(x[0])
##    plt.show()

##    tx = reduce_size(x, 0)
##    ty = reduce_size(y, 1)
##    tz = reduce_size(z, 1)
##    print('ti: ', tx.shape)
##    print('tl: ', ty.shape)
##    print('tsl: ', tz.shape)
##
##    np.savez('MINI_All_Fonts_(labeled by font and letter)' , tx, ty, tz)
##    print('FILES SAVED!!')

def test_each(file_name):
    x,y,z = load_data(file_name+'.npz')
    num = 0

    plt.imshow(x[0])
    plt.show()
    plt.imshow(x[7800])
    plt.show()


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ WORKING AREA $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
if __name__ == '__main__':

    run()
    test('MINI_PAIR_Fonts')
##    test_each('MINI_PAIR_Fonts_MINI_PAIR_Fonts')

    #operations = generate_permutaion_operation()
    #test_each()

''' THE WORKING CODE (END) '''
