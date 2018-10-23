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
from __future__ import print_function

import os
import numpy as np
from models import *
from utils import save_image
from tqdm import trange
from data_loader import data_loader

def to_nhwc(image, data_format):
    if data_format == 'NCHW':
        new_image = nchw_to_nhwc(image)
    else:
        new_image = image
    return new_image

def norm_img(image, data_format=None):
    image = image/127.5 - 1.
    if data_format:
        image = to_nhwc(image, data_format)
    return image

def denorm_img(norm, data_format):
    return tf.clip_by_value(to_nhwc((norm + 1)*127.5, data_format), 0, 255)

def switch_condition(condition):
    out = (-1*((2*condition)-1)+2)/2
    return out

class Trainer(object):
    def __init__(self, config):
        self.config         = config
        self.beta1          = config.beta1
        self.beta2          = config.beta2
        self.batch_size     = config.batch_size

        self.step           = tf.Variable(0, name='step', trainable=False)
        self.g_lr           = tf.Variable(config.g_lr, name='g_lr')
        self.z_lr           = tf.Variable(config.z_lr, name='d_lr')
        self.g_lr_update    = tf.assign(self.g_lr, tf.maximum(self.g_lr * 0.5, config.lr_lower_boundary), name='g_lr_update')
        self.z_lr_update    = tf.assign(self.z_lr, tf.maximum(self.z_lr * 0.5, config.lr_lower_boundary), name='z_lr_update')
        self.img,self.condition,self.p_num,self.n_num = data_loader(config)
        self.data_num       = self.p_num + self.n_num
        self.z_dim          = config.z_dim
        self.model_dir      = config.model_dir
        self.load_path      = config.load_path
        self.data_format    = config.data_format
        self.start_step     = 0
        self.log_step       = config.log_step
        self.max_step       = config.max_step
        self.lr_update_step = config.lr_update_step
        self.build_model()
        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.model_dir)
        sv = tf.train.Supervisor(logdir=self.model_dir,
                                is_chief=True,
                                saver=self.saver,
                                summary_op=None,
                                summary_writer=self.summary_writer,
                                save_model_secs=300,
                                global_step=self.step,
                                ready_for_local_init_op=None)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)

        self.sess = sv.prepare_or_wait_for_session(config=sess_config)

    def train(self):

        for step in trange(self.start_step, self.max_step):
            if step%(self.data_num/self.batch_size)==0:
                self.sess.run(self.update_latent)

            batch           = np.random.randint(0,self.data_num,size=[self.batch_size])
            self.sess.run(self.g_optim,feed_dict={self.INDEX:batch,self.X_REAL:self.img[batch],self.CONDITION:self.condition[batch]})
            self.sess.run(self.z_optim,feed_dict={self.INDEX:batch,self.X_REAL:self.img[batch],self.CONDITION:self.condition[batch]})

            if step % self.log_step == 0 :
                loss,result = self.sess.run([self.loss,self.summary_op],feed_dict={self.INDEX:batch,self.X_REAL:self.img[batch],self.CONDITION:self.condition[batch]})
                print("[{:6d}/{}] Loss: {:.6f} ". \
                    format(step, self.max_step,loss))
                self.summary_writer.add_summary(result, step)
                self.summary_writer.flush()


            if step % (self.log_step * 100 ) == 0:
                x       = self.generate(batch,self.model_dir, idx=step)
                x_real  = self.generate_real(batch,self.model_dir, idx=step)
            
            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run([self.g_lr_update, self.z_lr_update])
         

    def build_model(self):
        self.INDEX                  = tf.placeholder(tf.int32,[self.batch_size])
        self.X_REAL                 = tf.placeholder(tf.float32,shape=(self.batch_size,3,self.config.input_scale_size,self.config.input_scale_size))
        x_real                      = norm_img(self.X_REAL)   

        scale                       = tf.sqrt(float(self.z_dim))
        epsilon                     = 1e-07
        self.latent_z               = tf.Variable(tf.random_normal([self.data_num,self.z_dim])/scale, dtype=tf.float32)
        mean,variance               = tf.nn.moments(self.latent_z,[0,1])
        self.update_latent          = tf.assign(self.latent_z,tf.nn.batch_normalization(self.latent_z, mean, variance, 0, 1/scale, epsilon))

        self.CONDITION              = tf.placeholder(tf.float32,[None,1])
        look_up                     = tf.gather(self.latent_z,self.INDEX,axis=0)
        G,self.G_var                = Glo_Generator(look_up,self.CONDITION,self.batch_size,self.z_dim,reuse=False)

        optimizer                   = tf.train.AdamOptimizer
        g_optimizer, z_optimizer    = optimizer(self.g_lr,beta1=self.beta1,beta2=self.beta2), optimizer(self.z_lr,beta1=self.beta1,beta2=self.beta2)
        
        self.loss                   = tf.reduce_mean(tf.abs(x_real - G))
        
        self.g_optim                = g_optimizer.minimize(self.loss, var_list=self.G_var)
        self.z_optim                = z_optimizer.minimize(self.loss, var_list=self.latent_z)

        self.G                      = denorm_img(G, self.data_format)
        self.y                      = denorm_img(x_real, self.data_format)

        self.summary_op = tf.summary.merge([
            tf.summary.scalar('loss', self.loss)
        ])

    def test(self):
        dir_path_n      = self.config.n_data_dir
        dir_n           = os.listdir(dir_path_n)
        dir_n.sort()
        for i in range(len(dir_n)):
            dir_n[i]=dir_n[i].split('.jpg')
        path = self.model_dir+'_test/'
        if not os.path.exists(path):
            os.makedirs(path)
        
        for step in range(self.p_num/self.batch_size, self.data_num/self.batch_size):
            batch   = np.arange(self.batch_size) + (self.batch_size*step)
            x_condi = self.generate_condition(batch,path, idx=step,is_train=False,name=dir_n[step-self.p_num/self.batch_size][0])
            x_real  = self.generate_real(batch,path, idx=step,is_train=False,name=dir_n[step-self.p_num/self.batch_size][0])

    def generate(self,input,root_path=None, path=None, idx=None, is_train=True,name=None):
        
        x = self.sess.run(self.G,feed_dict={self.CONDITION:self.condition[input],self.INDEX:input})
        if path is None and is_train:
            path = os.path.join(root_path, '{}_G.png'.format(idx))
            save_image(x, path)
            print("[*] Samples saved: {}".format(path))
        return x
    
    def generate_condition(self,input,root_path=None, path=None, idx=None, is_train=True,name=None):
        # change conditional label
        x = self.sess.run(self.G,feed_dict={self.CONDITION:switch_condition(self.condition[input]),self.INDEX:input})
        if path is None and is_train:
            path = os.path.join(root_path, '{}_G_condition.png'.format(idx))
            save_image(x, path)
            print("[*] Samples saved: {}".format(path))
        else:
            path = os.path.join(root_path, '{}.png'.format(name))
            save_image(x, path, nrow=1, padding=0,is_train=is_train)
            print("[*] Samples saved: {}".format(path))
        return x


    def generate_real(self,input, root_path=None, path=None, idx=None, is_train=True,name=None):
        x = self.sess.run(self.y,feed_dict={self.X_REAL:self.img[input]})
        if path is None and is_train:
            path = os.path.join(root_path, '{}_G_real.png'.format(idx))
            save_image(x, path)
            print("[*] Samples saved: {}".format(path))
        else:
            path = os.path.join(root_path, '{}_real.png'.format(name))
            save_image(x, path, nrow=1, padding=0,is_train=is_train)
            print("[*] Samples saved: {}".format(path))
        return x
    

    
    
    








