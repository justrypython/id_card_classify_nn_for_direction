#encoding:UTF-8

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from convnets import convnet
from predict import predict
import datetime
        
def main():
    #-----------------------------------------------------------------
    # 1: Set some necessary parameters
    weights_path = 'model/v4_0_1_convnet_227_weights_epoch06_loss0.0012.h5'

    #-----------------------------------------------------------------
    # 2: Build the Keras model
    model = convnet('alexnet', weights_path=weights_path, heatmap=True)
    
    path = '/media/zhaoke/806602c3-72ac-4719-b178-abc72b3fa783/share/10000id_part/'
    dst_path = '/media/zhaoke/806602c3-72ac-4719-b178-abc72b3fa783/share/10000id_part_classified_2/'
    
    bad_case = ['/media/zhaoke/806602c3-72ac-4719-b178-abc72b3fa783/share/10000id_part/1YHK/3/3/943_songjing/1.jpg']
    
    starttime = datetime.datetime.now()
    print 'starttime: ', starttime
    
    imgs = []
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.jpg'):
                imgs.append(os.path.join(root, filename))
    
    cnt = 0
    for i in imgs:
        if i in bad_case:
            continue
        img = cv2.imread(i)
        if img is None:
            continue
        elif img.shape[0] < 227 or img.shape[1] < 227:
            continue
        #factor = min(img.shape[0]/720.0, img.shape[1]/1280.0)
        #reshape = (int(img.shape[1]/factor), int(img.shape[0]/factor))
        factor = min(img.shape[0]/227.0, img.shape[1]/227.0)
        reshape = (int(img.shape[1]/factor), int(img.shape[0]/factor))
        reshape = (max(227, reshape[0]), max(227, reshape[1]))
        raw_img = img.copy()
        img = cv2.resize(img, reshape)
        cnt += 1
        img = img[:, :, ::-1]
        result = model.predict(np.array([img]))
        result = predict(result)
        if result == 0:
            dst_img_path = os.path.join(dst_path, 'bgd', '%d_'%cnt+i[-i[::-1].find('/'):])
            cv2.imwrite(dst_img_path, img[:, :, ::-1])
        elif result == 1:
            dst_img_path = os.path.join(dst_path, 'neg', '%d_'%cnt+i[-i[::-1].find('/'):])
            cv2.imwrite(dst_img_path, img[:, :, ::-1])
        else:
            dst_img_path = os.path.join(dst_path, 'pos', '%d_'%cnt+i[-i[::-1].find('/'):])
            cv2.imwrite(dst_img_path, img[:, :, ::-1])
        if cnt % 20 == 0:
            print cnt
        
    endtime = datetime.datetime.now()
    print 'the total time is ', endtime - starttime
    
    

if __name__ == '__main__':
    main()
