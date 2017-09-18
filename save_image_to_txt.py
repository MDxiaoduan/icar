import numpy as np
import os
import cv2
from DeepLearning.Image import plot_images
from DeepLearning.python import shuffle_matrix
from _init_ import *

path = path_gray
CHARS = CHARS

len_data = len(os.listdir(path))

x = np.zeros((len_data, image_shape))
y = np.zeros((len_data, len(CHARS)))


def code_to_vec(code):                        # p是否是正确图片 code车牌号
    label = np.zeros((len(CHARS)))          #
    label[CHARS.index(code)] = 1.0          # index() 方法检测字符串中是否包含子字符串,将c表示的字符串位置处设为1
    return label

for index, filelist in enumerate(os.listdir(path)):                             # 0-9数据
    code_label = filelist.split('\\')[0][0]
    im = cv2.imread(path+filelist)
    # china
    # print(code_label)
    # print(im.shape)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_resize = cv2.resize(im_gray, (24, 24))                   # resize(weight, height)
    im_reshape = np.reshape(im_resize, (1, image_shape))
    x[index, :] = im_reshape
    y[index, :] = code_to_vec(code_label)

x_out, order = shuffle_matrix(x)
y_out, order = shuffle_matrix(y, order=order)

np.savetxt("data\\txt\\data_gray.txt", x_out,  delimiter=' ', fmt="%d", newline='\r\n')
np.savetxt("data\\txt\\label_gray.txt", y_out,  delimiter=' ', fmt="%d", newline='\r\n')

show = np.reshape(x_out[:16, :], (16, 24, 24))
plot_images(show, y_out[:16, :], show_color="gray")
