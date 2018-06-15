# Copyright 2017 The BEGAN-tensorflow Authors(Taehoon Kim). All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# MIT License
#
# Modifications copyright (c) 2018 Image & Vision Computing Lab, Institute of Information Science, Academia Sinica
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
slim = tf.contrib.slim

def Glo_Generator(x,condition,batch_size,z_dim,reuse):

    x_dim = z_dim
    condition_dim = 1


    with tf.variable_scope('G1',reuse=reuse) as g1:
        weights_g1 = {
            
            "w1_g1" : tf.get_variable("w1_g1",[x_dim + condition_dim , 4*4*1024]),   #4*4*1024
            "w2_g1" : tf.get_variable("w2_g1",[5, 5, 512, 1024]),   #8*8*512
            "w3_g1" : tf.get_variable("w3_g1",[5, 5, 256, 512]),    #16*16*256
            "w4_g1" : tf.get_variable("w4_g1",[5, 5, 128, 256]),    #32*32*128
            "w5_g1" : tf.get_variable("w5_g1",[5, 5, 3, 128]),      #64*64*3
            }

        biases_g1 = {
        
            "b1_g1" : tf.get_variable("b1_g1", [4*4*1024]),
            "b2_g1" : tf.get_variable("b2_g1", [512]),
            "b3_g1" : tf.get_variable("b3_g1", [256]),
            "b4_g1" : tf.get_variable("b4_g1", [128]),
            "b5_g1" : tf.get_variable("b5_g1", [3]),
            }
        
        
        g1_out1   = tf.add(tf.matmul(tf.concat([x,condition],1), weights_g1["w1_g1"]), biases_g1["b1_g1"])     #(16,8*8*512)
        g1_out1   = tf.reshape(g1_out1,[batch_size,4,4,1024])

        output_shape_g2 = tf.stack([batch_size, 8, 8, 512])
        g1_out2 = tf.nn.relu(slim.batch_norm(tf.add(deconv2d(g1_out1, weights_g1["w2_g1"], output_shape_g2), biases_g1["b2_g1"])))
        
        output_shape_g3 = tf.stack([batch_size, 16, 16, 256])
        g1_out3 = tf.nn.relu(slim.batch_norm(tf.add(deconv2d(g1_out2, weights_g1["w3_g1"], output_shape_g3), biases_g1["b3_g1"])))
        
        output_shape_g4 = tf.stack([batch_size, 32, 32, 128])
        g1_out4 = tf.nn.relu(slim.batch_norm(tf.add(deconv2d(g1_out3, weights_g1["w4_g1"], output_shape_g4), biases_g1["b4_g1"])))

        output_shape_g5 = tf.stack([batch_size, 64, 64, 3])
        g1_out5 = tf.nn.tanh(tf.add(deconv2d(g1_out4, weights_g1["w5_g1"], output_shape_g5), biases_g1["b5_g1"]))
        g1_out5 = tf.transpose(g1_out5,[0,3,1,2])

    variables_g1 = tf.contrib.framework.get_trainable_variables(g1)
    return g1_out5,variables_g1

def deconv2d(x, W, output_shape):
    return tf.nn.conv2d_transpose(x, W, output_shape, strides = [1, 2, 2, 1], padding = 'SAME')


def nchw_to_nhwc(x):
    return tf.transpose(x, [0, 2, 3, 1])

def nhwc_to_nchw(x):
    return tf.transpose(x, [0, 3, 1, 2])

