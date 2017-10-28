'''
Created on 2016-3-29

@author: IRISBEST
'''
import numpy
import sys,os,time,random
from RepresentationLayer import RepresentationLayer
from Constants import *
import Eval
import FileUtil
import Corpus

from keras import models
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, Input, merge
from keras.layers.merge import Concatenate 
from keras.layers import LSTM, SimpleRNN
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling1D
from keras.layers.core import Flatten
from keras.optimizers import RMSprop, SGD, Adam, Adadelta, Adagrad
from keras.layers.wrappers import Bidirectional
from keras import backend as K
#from keras.utils.visualize_util import plot
import theano
# theano.config.device = 'gpu0'


if __name__ == '__main__':


    dir = '/home/dlut/research/RE/'
    max_len = 80
    word_vec_dim = 50
    position_vec_dim = 10
    epoch_size = 100
    rep = RepresentationLayer(wordvec_file=dir + 'data/wordemb_d50_slim',
                                frequency = 15000, max_sent_len=max_len)


    word = Input(shape=(max_len,), dtype='int32', name='word')
    distance_e1 = Input(shape=(max_len,), dtype='int32', name='distance_e1')
    distance_e2 = Input(shape=(max_len,), dtype='int32', name='distance_e2')
    
    word_emb = Embedding(rep.vec_table.shape[0], rep.vec_table.shape[1],
                         weights = [rep.vec_table], mask_zero=True, input_length=max_len)
    #, mask_zero=True
    position_emb = Embedding(max_len * 2 + 1, position_vec_dim, mask_zero=True, input_length=max_len)
    
    word_vec = word_emb(word)
    distance1_vec = position_emb(distance_e1)
    distance2_vec = position_emb(distance_e2)


    # generate the input vector for LSTM
    concatenated  = Concatenate()([word_vec, distance1_vec, distance2_vec])
    dropouted = Dropout(0.5)(concatenated)

    lstm = LSTM(100, activation='tanh')(dropouted)
    dense = Dense(100, activation='tanh')(lstm)
    
#    cnn = Convolution1D(nb_filter=100, filter_length=3, activation='tanh')(concat_vec)
#    cnn = Convolution1D(nb_filter=100, filter_length=3, activation='tanh')(cnn)
#    flattened = Flatten()(cnn)
#    dense = Dense(100, activation='tanh')(flattened)
    
    predict = Dense(2, activation='softmax')(dense)
    model = Model(inputs=[word, distance_e1, distance_e2], outputs=predict)

    opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
#    opt = Adagrad(lr=0.01, epsilon=1e-06)
#    opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
#    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#    opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=opt)
#     model.summary()
    
#     learning_phase=K.learning_phase();
#     intermediate_tensor_function = K.function([word, distance_e1, distance_e2,learning_phase], [lstm])

    # traing the model
    train_instances = [line.strip() for line in open(dir + 'data/ddi_2013_drugbank-train.token')]
    dev_instances = [line.strip() for line in open(dir + 'data/ddi_2013_drugbank-dev.token')]
    label_array_t, word_array_t, dis_e1_array_t, dis_e2_array_t = rep.represent_instances(train_instances)
    label_array_d, word_array_d, dis_e1_array_d, dis_e2_array_d = rep.represent_instances(dev_instances)
     
    best_f = 0
    for epoch in xrange(epoch_size):
        print 'running the epoch:', (epoch + 1)
        model.fit([word_array_t, dis_e1_array_t, dis_e2_array_t],label_array_t, batch_size=128, epochs=1)
        answer_array_d = model.predict([word_array_d, dis_e1_array_d, dis_e2_array_d], batch_size=128)
        current_f = Eval.eval_mulclass(label_array_d, answer_array_d)
        if current_f > best_f:
            print 'New Best F-score'
            best_f = current_f
            model.save_weights('./lstm_model.h5')
        else:
            print 'Lower than before'
          
#         inter_tensors =  intermediate_tensor_function([word_array, dis_e1_array, dis_e2_array,0])
#         print inter_tensors[0]
    
    # predict the test data using the model
    test_instances = [line.strip() for line in open(dir + 'data/ddi_2013_drugbank-test.token')]
    label_array_test, word_array_test, dis_e1_array_test, dis_e2_array_test = rep.represent_instances(test_instances)
    print 'load the model'
    model.load_weights('./lstm_model.h5')
    print 'predicting ...'
    answer_array_test = model.predict([word_array_test, dis_e1_array_test, dis_e2_array_test], batch_size=128)
    Eval.eval_mulclass(label_array_test, answer_array_test)
 
