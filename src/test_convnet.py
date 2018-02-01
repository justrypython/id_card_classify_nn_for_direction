#encoding:UTF-8

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from convnets import convnet
import icdar


def test_noheap():
    #-----------------------------------------------------------------
    # 1: Set some necessary parameters
    weights_path = 'model/v2_0_convnet_227_weights_epoch05_loss0.0033.h5'
    size = 227
    labels = {'0': 0,
              '1': 1,
              '2': 2,
              '3': 3,
              '4': 4,
              '5': 5,
              '6': 6,
              '7': 7,
              '8': 8,
              '9': 9,
              '10': 10,
              '15': 11,
              '16': 12}

    #-----------------------------------------------------------------
    # 2: Build the Keras model
    model = convnet('alexnet', weights_path=weights_path, heatmap=False)
    model.save_weights('test_model/dense.h5')
    
    #-----------------------------------------------------------------
    # 5: Create the validation set batch generator
    data_generator = icdar.generator(input_size=227, batch_size=1, labels=labels, vis=False)
    cnt = 0
    right = 0
    while True:
        cnt += 1
        X, y_true = next(data_generator)
        y_pred = model.predict(X)
        y_true_label = np.argmax(y_true)
        y_pred_label = np.argmax(y_pred)
        if y_pred_label == 0:
            print 'y_pred is background'
        elif y_pred_label == 1:
            print 'y_pred is negedge'
        else:
            print 'y_pred is posedge'
        if y_true_label == y_pred_label:
            print True
            right += 1
        else:
            print False
        print 'the accuracy is %f'%(float(right)/cnt)
        
def test_heap_one():
    #-----------------------------------------------------------------
    # 1: Set some necessary parameters
    weights_path = 'model/convnet_227_weights_epoch02_loss0.0030.h5'   
    size = 227
    labels = {'0': 0,
              '1': 1,
              '2': 2,
              '3': 3,
              '4': 4,
              '5': 5,
              '6': 6,
              '7': 7,
              '8': 8,
              '9': 9,
              '10': 10,
              '15': 11,
              '16': 12}

    #-----------------------------------------------------------------
    # 2: Build the Keras model
    model = convnet('alexnet', weights_path=weights_path, heatmap=True)
    model.save_weights('test_model/conv.h5')
    
    #-----------------------------------------------------------------
    # 5: Create the validation set batch generator
    data_generator = icdar.generator(input_size=227, batch_size=1, labels=labels, vis=False)
    cnt = 0
    right = 0
    while True:
        cnt += 1
        X, y_true = next(data_generator)
        y_pred = model.predict(X)
        y_true_label = np.argmax(y_true)
        y_pred_label = np.argmax(y_pred)
        if y_pred_label == 0:
            print 'y_pred is background'
        elif y_pred_label == 1:
            print 'y_pred is negedge'
        else:
            print 'y_pred is posedge'
        if y_true_label == y_pred_label:
            print True
            right += 1
        else:
            print False
        print 'the accuracy is %f'%(float(right)/cnt)
        
def test_compare():
    #-----------------------------------------------------------------
    # 1: Set some necessary parameters
    heap_weights_path = 'model/convnet_227_weights_epoch02_loss0.0030.h5'  
    nohp_weights_path = 'model/convnet_227_weights_epoch02_loss0.0030.h5'    
    size = 227
    labels = {'0': 0,
              '1': 1,
              '2': 2,
              '3': 3,
              '4': 4,
              '5': 5,
              '6': 6,
              '7': 7,
              '8': 8,
              '9': 9,
              '10': 10,
              '15': 11,
              '16': 12}
    
    #-----------------------------------------------------------------
    # 5: Create the validation set batch generator
    data_generator = icdar.generator(input_size=227, batch_size=1, labels=labels, vis=False)
    with tf.Session() as sess:
        nohp_model = convnet('alexnet', weights_path=nohp_weights_path, heatmap=False)
        heap_model = convnet('alexnet', weights_path=heap_weights_path, heatmap=True)
        X, y_true = next(data_generator)
        nohp_y_pred = nohp_model.predict(X)
        heap_y_pred = heap_model.predict(X)
        graph = tf.get_default_graph()
        ops = [v for v in graph.get_operations()]
        plcers = [i for i in ops if i.type=='Placeholder']
        pool0 = graph.get_tensor_by_name('convpool_5/MaxPool:0')
        pool1 = graph.get_tensor_by_name('convpool_5_1/MaxPool:0')
        pool2 = graph.get_tensor_by_name('convpool_5_2/MaxPool:0')
        a, b, c = sess.run([pool0, pool1, pool2], feed_dict={'input_1:0':X, 'input_2:0':X, 'input_3:0':X})
        dense0 = graph.get_tensor_by_name('dense_1/Relu:0')
        dense1 = graph.get_tensor_by_name('dense_1_1/Relu:0')
        dense2 = graph.get_tensor_by_name('dense_1_2/Relu:0')
        a, b, c = sess.run([dense0, dense1, dense2], feed_dict={'input_1:0':X, 'input_2:0':X, 'input_3:0':X})
        dense0 = graph.get_tensor_by_name('dense_2/Relu:0')
        dense1 = graph.get_tensor_by_name('dense_2_1/Relu:0')
        dense2 = graph.get_tensor_by_name('dense_2_2/Relu:0')
        a, b, c = sess.run([dense0, dense1, dense2], feed_dict={'input_1:0':X, 'input_2:0':X, 'input_3:0':X,
                                                                'dropout_1/keras_learning_phase:0':False})
        dense0 = graph.get_tensor_by_name('dense_3/BiasAdd:0')
        dense1 = graph.get_tensor_by_name('dense_3_1/BiasAdd:0')
        dense2 = graph.get_tensor_by_name('dense_3_2/BiasAdd:0')
        a, b, c = sess.run([dense0, dense1, dense2], feed_dict={'input_1:0':X, 'input_2:0':X, 'input_3:0':X,
                                                                'dropout_1/keras_learning_phase:0':False})
        dense0 = graph.get_tensor_by_name('dense_4/BiasAdd:0')
        dense1 = graph.get_tensor_by_name('dense_4_1/BiasAdd:0')
        dense2 = graph.get_tensor_by_name('dense_4_2/BiasAdd:0')
        a, b, c = sess.run([dense0, dense1, dense2], feed_dict={'input_1:0':X, 'input_2:0':X, 'input_3:0':X,
                                                                'dropout_1/keras_learning_phase:0':False})
        dense0 = graph.get_tensor_by_name('softmax/Softmax:0')
        dense1 = graph.get_tensor_by_name('softmax_1/Softmax:0')
        dense2 = graph.get_tensor_by_name('softmax_2/Reshape_1:0')
        a, b, c = sess.run([dense0, dense1, dense2], feed_dict={'input_1:0':X, 'input_2:0':X, 'input_3:0':X,
                                                                'dropout_1/keras_learning_phase:0':False})
        print 'end'

if __name__ == '__main__':
    test_heap_one()