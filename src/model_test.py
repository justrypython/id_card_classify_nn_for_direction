#encoding:UTF-8

import os
import numpy as np
import tensorflow as tf
import h5py
from keras.optimizers import SGD
from convnets import convnet

def main():
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model = convnet('alexnet', heatmap=False)
    model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
    
    with h5py.File('model/alexnet_weights.h5', 'r') as f:
        names = f.keys()
        for i in names:
            print i
            for j in f[i].keys():
                print f[i][j].shape
        print 'print h5 end'
    layers = model.layers
    for i in layers:
        print i.name
        for j in i.get_weights():
            print j.shape
    print 'print model end'
    for i in range(5):
        print '\n'
    f = h5py.File('model/alexnet_weights.h5', 'r')
    for i in layers:
        name = i.name
        if 'conv_' in name:
            if len(i.get_weights()) > 0:
                print 'load %s weights'%name
                w = f[name][name+'_W']
                b = f[name][name+'_b']
                w = np.transpose(w, [2, 3, 1, 0])
                i.set_weights([w, b])
            else:
                print 'layers %s has no weights'%name
        elif 'dense_' in name and name != 'dense_4':
            print 'load %s weights'%name
            w = f[name][name+'_W']
            b = f[name][name+'_b']
            i.set_weights([w, b])
    print 'load end!!!'
    print 'save weights'
    model.save_weights('model/tf_alexnet_weights.h5')

def main_less_dense():
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model = convnet('alexnet_less_dense', heatmap=False)
    model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
    
    with h5py.File('model/alexnet_weights.h5', 'r') as f:
        names = f.keys()
        for i in names:
            print i
            for j in f[i].keys():
                print f[i][j].shape
        print 'print h5 end'
    layers = model.layers
    for i in layers:
        print i.name
        for j in i.get_weights():
            print j.shape
    print 'print model end'
    for i in range(5):
        print '\n'
    f = h5py.File('model/alexnet_weights.h5', 'r')
    for i in layers:
        name = i.name
        if 'conv_' in name:
            if len(i.get_weights()) > 0:
                print 'load %s weights'%name
                w = f[name][name+'_W']
                b = f[name][name+'_b']
                w = np.transpose(w, [2, 3, 1, 0])
                i.set_weights([w, b])
            else:
                print 'layers %s has no weights'%name
        #elif 'dense_' in name and name != 'dense_4':
            #print 'load %s weights'%name
            #w = f[name][name+'_W']
            #b = f[name][name+'_b']
            #i.set_weights([w, b])
    print 'load end!!!'
    print 'save weights'
    model.save_weights('model/tf_alexnet_less_dense_weights.h5')

def main_direction():
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model = convnet('alexnet', heatmap=False)
    model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
    
    with h5py.File('model/alexnet_weights.h5', 'r') as f:
        names = f.keys()
        for i in names:
            print i
            for j in f[i].keys():
                print f[i][j].shape
        print 'print h5 end'
    layers = model.layers
    for i in layers:
        print i.name
        for j in i.get_weights():
            print j.shape
    print 'print model end'
    for i in range(5):
        print '\n'
    f = h5py.File('model/alexnet_weights.h5', 'r')
    for i in layers:
        name = i.name
        if 'conv_' in name:
            if len(i.get_weights()) > 0:
                print 'load %s weights'%name
                w = f[name][name+'_W']
                b = f[name][name+'_b']
                w = np.transpose(w, [2, 3, 1, 0])
                i.set_weights([w, b])
            else:
                print 'layers %s has no weights'%name
        elif 'dense_' in name and name != 'dense_4':
            print 'load %s weights'%name
            w = f[name][name+'_W']
            b = f[name][name+'_b']
            i.set_weights([w, b])
    print 'load end!!!'
    print 'save weights'
    model.save_weights('model/tf_alexnet_direction_weights.h5')
    
def compare_weights():
    dense_file = h5py.File('test_model/dense.h5', 'r')
    conv_file = h5py.File('test_model/conv.h5', 'r')
    for i in dense_file.keys():
        print i, 'is in dense'
        if i in conv_file.keys():
            print '%s also in conv'%i
        else:
            print '%s not in conv'%i
    dense_w1 = dense_file['dense_1']['dense_1']['kernel:0']
    dense_b1 = dense_file['dense_1']['dense_1']['bias:0']
    conv_w1 = conv_file['dense_1']['dense_1_1']['kernel:0']
    conv_b1 = conv_file['dense_1']['dense_1_1']['bias:0']
    
    a = conv_w1.value
    b = dense_w1.value
    a = a.reshape(b.shape)
    print np.all(a==b)
    
    dense_w2 = dense_file['dense_2']['dense_2']['kernel:0']
    dense_b2 = dense_file['dense_2']['dense_2']['bias:0']
    conv_w2 = conv_file['dense_2']['dense_2_1']['kernel:0']
    conv_b2 = conv_file['dense_2']['dense_2_1']['bias:0']
    
    a = conv_w2.value
    b = dense_w2.value
    a = a.reshape(b.shape)
    print np.all(a==b)
    
    dense_w3 = dense_file['dense_3']['dense_3']['kernel:0']
    dense_b3 = dense_file['dense_3']['dense_3']['bias:0']
    conv_w3 = conv_file['dense_3']['dense_3_1']['kernel:0']
    conv_b3 = conv_file['dense_3']['dense_3_1']['bias:0']
    
    a = conv_w3.value
    b = dense_w3.value
    a = a.reshape(b.shape)
    print np.all(a==b)
    
    dense_w4 = dense_file['dense_4']['dense_4']['kernel:0']
    dense_b4 = dense_file['dense_4']['dense_4']['bias:0']
    conv_w4 = conv_file['dense_4']['dense_4_1']['kernel:0']
    conv_b4 = conv_file['dense_4']['dense_4_1']['bias:0']
    
    a = conv_w4.value
    b = dense_w4.value
    a = a.reshape(b.shape)
    print np.all(a==b)
    
    dense_file.close()
    conv_file.close()
    
def set_weights():
    dense_file = h5py.File('test_model/dense.h5', 'r')
    conv_file = h5py.File('test_model/conv.h5', 'r+')
    for i in dense_file.keys():
        print i, 'is in dense'
        if i in conv_file.keys():
            print '%s also in conv'%i
        else:
            print '%s not in conv'%i
    dense_w1 = dense_file['dense_1']['dense_1']['kernel:0']
    dense_b1 = dense_file['dense_1']['dense_1']['bias:0']
    conv_w1 = conv_file['dense_1']['dense_1_1']['kernel:0'].value
    conv_b1 = conv_file['dense_1']['dense_1_1']['bias:0']
    
    conv_file.__delitem__('dense_1/dense_1_1/kernel:0')
    conv_file['dense_1']['dense_1_1']['kernel:0'] = dense_w1.value.reshape(conv_w1.shape)
    
    dense_w2 = dense_file['dense_2']['dense_2']['kernel:0']
    dense_b2 = dense_file['dense_2']['dense_2']['bias:0']
    conv_w2 = conv_file['dense_2']['dense_2_1']['kernel:0'].value
    conv_b2 = conv_file['dense_2']['dense_2_1']['bias:0']

    conv_file.__delitem__('dense_2/dense_2_1/kernel:0')
    conv_file['dense_2']['dense_2_1']['kernel:0'] = dense_w2.value.reshape(conv_w2.shape)    
    
    dense_w3 = dense_file['dense_3']['dense_3']['kernel:0']
    dense_b3 = dense_file['dense_3']['dense_3']['bias:0']
    conv_w3 = conv_file['dense_3']['dense_3_1']['kernel:0'].value
    conv_b3 = conv_file['dense_3']['dense_3_1']['bias:0']

    conv_file.__delitem__('dense_3/dense_3_1/kernel:0')
    conv_file['dense_3']['dense_3_1']['kernel:0'] = dense_w3.value.reshape(conv_w3.shape)  
    
    dense_w4 = dense_file['dense_4']['dense_4']['kernel:0']
    dense_b4 = dense_file['dense_4']['dense_4']['bias:0']
    conv_w4 = conv_file['dense_4']['dense_4_1']['kernel:0'].value
    conv_b4 = conv_file['dense_4']['dense_4_1']['bias:0']

    conv_file.__delitem__('dense_4/dense_4_1/kernel:0')
    conv_file['dense_4']['dense_4_1']['kernel:0'] = dense_w4.value.reshape(conv_w4.shape)
    
    dense_file.close()
    conv_file.close()
        
if __name__ == '__main__':
    #main_less_dense()
    main_direction()