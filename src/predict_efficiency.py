#encoding:UTF-8

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from convnets import convnet
import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = ''


def predict(result, thres=0.7):
    #result = np.squeeze(result)
    #if len(result.shape) == 2:
        #result = np.sum(result, axis=0)
    #return np.argmax(result)
    #-----------2nd generation----------------------------------------
    #return np.argmax(result) % 3
    #-----------1st generation----------------------------------------
    result[result>thres] = 1
    result[result<=thres] = 0
    result = np.argmax(result, axis=-1)
    neg = len(result[result==1])
    pos = len(result[result==2])
    if pos == neg == 0:
        return 0
    elif pos > neg:
        return 2
    else:
        return 1
        
def main():
    #-----------------------------------------------------------------
    # 1: Set some necessary parameters
    #weights_path = 'model/v4_0_1_convnet_227_weights_epoch06_loss0.0012.h5'
    weights_path = 'model/v4_0_3_convnet_227_weights_epoch04_loss0.0004.h5'

    #-----------------------------------------------------------------
    # 2: Build the Keras model
    model = convnet('alexnet', weights_path=weights_path, heatmap=True)
    model1 = convnet('alexnet_less_dense', heatmap=True)
    
    posedge_path = '/home/zhaoke/justrypython/ks_idcard_ocr/testimg/card_bat/'
    negedge_path = '/home/zhaoke/justrypython/ks_idcard_ocr/testimg/neg_imgs/'
    background_path = '/home/zhaoke/gtest/ADEChallengeData2016/images/training2w/'
    
    step = 355.0
    print '-----------%f------------'%step
    starttime = datetime.datetime.now()
    print 'start time is ', starttime
    
    pos_cnt = 0
    pos_rgt = 0
    for i in os.listdir(posedge_path):
        img = cv2.imread(posedge_path+i)
        img = img[:, :, ::-1]
        factor = min(img.shape[0]/step, img.shape[1]/step)
        reshape = (int(img.shape[1]/factor), int(img.shape[0]/factor))
        img = cv2.resize(img, reshape)
        if img.shape[0] < 227 or img.shape[1] < 227:
            continue
        result = model.predict(np.array([img]))
        result = predict(result)
        pos_cnt += 1
        if result == 2:
            pos_rgt += 1
        else:
            pass
            #cv2.imwrite('results/pos/%d.jpg'%pos_cnt, img[:, :, ::-1])
    
    neg_cnt = 0
    neg_rgt = 0
    for i in os.listdir(negedge_path):
        img = cv2.imread(negedge_path+i)
        img = img[:, :, ::-1]
        factor = min(img.shape[0]/step, img.shape[1]/step)
        reshape = (int(img.shape[1]/factor), int(img.shape[0]/factor))
        img = cv2.resize(img, reshape)
        if img.shape[0] < 227 or img.shape[1] < 227:
            continue
        result = model.predict(np.array([img]))
        result = predict(result)
        neg_cnt += 1
        if result == 1:
            neg_rgt += 1
        else:
            pass
            #cv2.imwrite('results/neg/%d.jpg'%neg_cnt, img[:, :, ::-1])
    
    bck_cnt = 0
    bck_rgt = 0
    for i in os.listdir(background_path):
        img = cv2.imread(background_path+i)
        img = img[:, :, ::-1]
        factor = min(img.shape[0]/step, img.shape[1]/step)
        reshape = (int(img.shape[1]/factor), int(img.shape[0]/factor))
        img = cv2.resize(img, reshape)
        if img.shape[0] < 227 or img.shape[1] < 227:
            continue
        result = model.predict(np.array([img]))
        result = predict(result)
        bck_cnt += 1
        if result == 0:
            bck_rgt += 1
        else:
            pass
            #cv2.imwrite('results/bgd/%d.jpg'%bck_cnt, img[:, :, ::-1])
        if bck_cnt > 500:
            break
        
    print 'the posedge rate is %.3f'%(float(pos_rgt)/pos_cnt)
    print 'the negedge rate is %.3f'%(float(neg_rgt)/neg_cnt)
    print 'the background rate is %.3f'%(float(bck_rgt)/bck_cnt)
    print 'the total rate is %.3f'%(float(pos_rgt+neg_rgt+bck_rgt)/(pos_cnt+neg_cnt+bck_cnt))

    endtime = datetime.datetime.now()
    print 'the total time is ', endtime - starttime
    

    step = 355.0
    print '-----------%f------------'%step
    starttime = datetime.datetime.now()
    print 'start time is ', starttime
    
    pos_cnt = 0
    pos_rgt = 0
    for i in os.listdir(posedge_path):
        img = cv2.imread(posedge_path+i)
        img = img[:, :, ::-1]
        factor = min(img.shape[0]/step, img.shape[1]/step)
        reshape = (int(img.shape[1]/factor), int(img.shape[0]/factor))
        img = cv2.resize(img, reshape)
        if img.shape[0] < 227 or img.shape[1] < 227:
            continue
        result = model1.predict(np.array([img]))
        result = predict(result)
        pos_cnt += 1
        if result == 2:
            pos_rgt += 1
        else:
            pass
            #cv2.imwrite('results/pos/%d.jpg'%pos_cnt, img[:, :, ::-1])
    
    neg_cnt = 0
    neg_rgt = 0
    for i in os.listdir(negedge_path):
        img = cv2.imread(negedge_path+i)
        img = img[:, :, ::-1]
        factor = min(img.shape[0]/step, img.shape[1]/step)
        reshape = (int(img.shape[1]/factor), int(img.shape[0]/factor))
        img = cv2.resize(img, reshape)
        if img.shape[0] < 227 or img.shape[1] < 227:
            continue
        result = model1.predict(np.array([img]))
        result = predict(result)
        neg_cnt += 1
        if result == 1:
            neg_rgt += 1
        else:
            pass
            #cv2.imwrite('results/neg/%d.jpg'%neg_cnt, img[:, :, ::-1])
    
    bck_cnt = 0
    bck_rgt = 0
    for i in os.listdir(background_path):
        img = cv2.imread(background_path+i)
        img = img[:, :, ::-1]
        factor = min(img.shape[0]/step, img.shape[1]/step)
        reshape = (int(img.shape[1]/factor), int(img.shape[0]/factor))
        img = cv2.resize(img, reshape)
        if img.shape[0] < 227 or img.shape[1] < 227:
            continue
        result = model1.predict(np.array([img]))
        result = predict(result)
        bck_cnt += 1
        if result == 0:
            bck_rgt += 1
        else:
            pass
            #cv2.imwrite('results/bgd/%d.jpg'%bck_cnt, img[:, :, ::-1])
        if bck_cnt > 500:
            break
        
    print 'the posedge rate is %.3f'%(float(pos_rgt)/pos_cnt)
    print 'the negedge rate is %.3f'%(float(neg_rgt)/neg_cnt)
    print 'the background rate is %.3f'%(float(bck_rgt)/bck_cnt)
    print 'the total rate is %.3f'%(float(pos_rgt+neg_rgt+bck_rgt)/(pos_cnt+neg_cnt+bck_cnt))

    endtime = datetime.datetime.now()
    print 'the total time is ', endtime - starttime    
    
    print 'end'
    
    

if __name__ == '__main__':
    main()
