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
data_file_size = str(data_file_size - 16) + 'B'
label_file_size = str(label_file_size - 8) + 'B'

data_buffer = open(train_images_path, 'rb').read()      # 以二进制形式打开
label_buffer = open(train_labels_path, 'rb').read()

magic, num_images, num_rows, num_columns = struct.unpack_from('>IIII', data_buffer, 0)
datas = struct.unpack_from('>'+data_file_size, data_buffer, struct.calcsize('>III'))

label_magic, num_labels = struct.unpack_from('>II',label_buffer, 0)
labels = struct.unpack_from('>'+label_file_size, label_buffer, struct.calcsize('>II'))

datas_array = np.asarray(datas).astype(float)
images = datas_array.reshape(num_images, 1, num_rows, num_columns)

labels_array = np.asarray(labels).astype(np.int64)


data_root = './images'
for i in range(10):
    file_name = os.path.join(data_root, str(i))
    if not os.path.exists(file_name):
        os.makedirs(file_name)

for j in tqdm(range(num_labels)):
    label = labels[j]
    img_path = os.path.join(data_root, str(label), str(j)+'.jpg')
    misc.toimage(images[j, 0, 0:28, 0:28]).save(img_path)