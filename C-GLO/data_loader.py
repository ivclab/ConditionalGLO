# MIT Licens
#
# Copyright (c) 2018 Image & Vision Computing Lab, Institute of Information Science, Academia Sinica
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
import numpy as np
import tensorflow as tf
from scipy.misc import imresize
import os
import skimage.io as io
import warnings
from tqdm import trange
warnings.simplefilter(action='ignore', category=FutureWarning)

def data_loader(config):
    dir_path_p      = config.p_data_dir
    dir_p           = os.listdir(dir_path_p)
    dir_p.sort()

    dir_path_n      = config.n_data_dir
    dir_n           = os.listdir(dir_path_n)
    dir_n.sort()

    p_num           = len(dir_p) #30496
    n_num           = len(dir_n) #7664
    size            = config.input_scale_size
    img             = []
    label           = []
    print('Loading data...')
    print('Loading positive data')
    for i in trange(p_num):
        if config.is_train:
            data = io.imread(dir_path_p + os.sep + dir_p[i])
            data = imresize(data,[size,size,data.shape[2]])
            img.append(data)
        label.append(1)

    print('Loading negative data')
    for i in trange(n_num):
        data = io.imread(dir_path_n + os.sep + dir_n[i])
        data = imresize(data,[size,size,data.shape[2]])
        img.append(data)
        label.append(0)

    label   = np.array(label,dtype = np.float32).reshape([-1,1])
    if config.is_train:
        img     = nhwc_to_nchw(np.array(img,dtype = np.float32))

    return img,label,p_num,n_num



def nhwc_to_nchw(x):
    return np.transpose(x, [0, 3, 1, 2])
