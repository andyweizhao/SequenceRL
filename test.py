# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 21:24:58 2017

@author: nlp
"""

import numpy as np
import tensorflow as tf
import random
from beamlstm import BeamLstm
from lstm import BasicLstm
from rllstm import RLLstm

class BatchGenerator(object):
    """Generate and hold batches."""
    def __init__(self, text, batch_size,maxlen):
      self._text = text
      self._batch_size = batch_size    
      self._cursor = 0
      self._maxlen = maxlen
      
    def next(self):
      """Generate a single batch from the current cursor position in the data."""
      batch = np.zeros(shape=(self._batch_size,self._maxlen), dtype=np.int32)
      for b in range(self._batch_size):    
          batch[b] = self._text[self._cursor + b,:]
      self._cursor += self._batch_size
      return batch
      
#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH = 20 # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 120 # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 5

#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 800
positive_file = 'save/real_data.txt'
negative_file = 'save/generator_sample.txt'
eval_file = 'save/eval_file.txt'
generated_num = 1000

random.seed(SEED)
np.random.seed(SEED)
vocab_size = 5000
assert START_TOKEN == 0    

with open('save/real_data.txt') as f:
    real_data = f.read().split('\n')
real_data = np.array([real_data[i].split() for i in range(len(real_data)-2)],dtype=np.int32)

R_batch = BatchGenerator(real_data, BATCH_SIZE , 20)

lstm = RLLstm(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN,grad_clip=5.0,
                 learning_rate=0.01,is_sample=True)

sess = tf.InteractiveSession()
tf.initialize_all_variables().run()

batch_real_data = R_batch.next()  

_, g_loss = lstm.pretrain_step(sess, batch_real_data)    
samples = lstm.generate(sess)

rewards = lstm.get_reward(sess,samples,sample_cnt=5)
#_, g_loss2 = lstm.unsupervised_train_step(sess,rewards)

#lstm.save_model(sess,os.path.join('Model', 'model'),global_step=1)
#lstm.restore_model(sess,'Model/model-1.pkl')
    

#_, g_loss = G_network.pretrain_step(sess, batch_real_data)         
#samples = np.hstack(G_network.generate(sess))
#a = np.hstack(G_network.getReward(sess,samples))