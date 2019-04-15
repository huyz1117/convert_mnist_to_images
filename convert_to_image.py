# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 19:01:57 2019

@author: huyuanzheng3
"""

import os
import struct
import numpy as np
from tqdm import tqdm
from scipy import misc

train_images_path = './mnist/train-images.idx3-ubyte'
train_labels_path = './mnist/train-labels.idx1-ubyte'

data_file_size = 47040016
label_file_size = 60008

'''
struct.unpack_from(fmt, file, offset):
该函数将可以将缓冲区buffer中的内容按照指定的格式fmt，从偏移量为offset的位置开始进行读取。
返回的是一个对应的元组tuple，一般的使用场景是从一个二进制或者其他的文件中读取内容进行解析操作
'''

# 以二进制格式打开文件，读取内容放在缓冲区
data_buffer = open(train_images_path, 'rb').read()      
label_buffer = open(train_labels_path, 'rb').read()

# 内容解析格式，其中B表示一个字节，>表示大端法则，data_file_size表示多少个字节
data_fmt = '>' + str(data_file_size - 16) + 'B'
label_fmt = '>' + str(label_file_size - 8) + 'B'

# magic, num_images, num_rows, num_columns = struct.unpack_from('>IIII', data_buffer, 0)
datas = struct.unpack_from(data_fmt, data_buffer, struct.calcsize('>III'))

# label_magic, num_labels = struct.unpack_from('>II',label_buffer, 0)
labels = struct.unpack_from(label_fmt, label_buffer, struct.calcsize('>II'))

datas_array = np.asarray(datas).astype(float)
# images = datas_array.reshape(num_images, 1, num_rows, num_columns)
images = datas_array.reshape(60000, 1, 28, 28)

labels_array = np.asarray(labels).astype(np.int64)


data_root = './image'
for i in range(10):
    file_name = os.path.join(data_root, str(i))
    if not os.path.exists(file_name):
        os.makedirs(file_name)

for j in tqdm(range(60000)):
    label = labels[j]
    img_path = os.path.join(data_root, str(label), str(j)+'.jpg')
    misc.toimage(images[j, 0, 0:28, 0:28]).save(img_path)