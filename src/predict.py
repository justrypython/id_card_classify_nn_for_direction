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

#os.environ['CUDA_VISIBLE_DEVICES'] = ''

def figure_result(img, result):
    fig, axs = plt.subplots(3, 2)
    axs[0, 0].imshow(img)
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    axs[0, 1].imshow(result[:, :, 0])
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    axs[1, 0].imshow(result[:, :, 1])
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])
    axs[1, 1].imshow(result[:, :, 2])
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    axs[2, 0].imshow(result[:, :, 3])
    axs[2, 0].set_xticks([])
    axs[2, 0].set_yticks([])
    axs[2, 1].imshow(result[:, :, 4])
    axs[2, 1].set_xticks([])
    axs[2, 1].set_yticks([])
    plt.tight_layout()
    plt.show()
    plt.close()

def predict(result, thres=0.8):
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
    up = len(result[result==1])
    right = len(result[result==2])
    down = len(result[result==3])
    left = len(result[result==4])
    maxnum = max([up, right, down, left])
    if up == right == down == left == 0:
        return 0
    elif up == maxnum:
        return 1
    elif right == maxnum:
        return 2
    elif down == maxnum:
        return 3
    else:
        return 4
        
def main():
    #-----------------------------------------------------------------
    # 1: Set some necessary parameters
    #weights_path = 'model/v4_0_1_convnet_227_weights_epoch06_loss0.0012.h5'
    #weights_path = 'model/convnet_direction_227_weights_epoch02_loss0.0835.h5'
    #weights_path = 'model/convnet_direction_227_weights_epoch03_loss0.0820.h5'
    #weights_path = 'model/convnet_direction_227_weights_epoch01_loss0.0760.h5'
    #weights_path = 'model/convnet_direction_227_weights_epoch02_loss0.0741.h5'
    #weights_path = 'model/convnet_direction_227_weights_epoch04_loss0.0673.h5'
    #weights_path = 'model/convnet_direction_227_weights_epoch10_loss0.0524.h5'
    weights_path = 'model/convnet_direction_227_weights_epoch06_loss0.0499.h5'

    #-----------------------------------------------------------------
    # 2: Build the Keras model
    model = convnet('alexnet', weights_path=weights_path, heatmap=True)
    
    #image_path = '/home/share/model_share/ks_nature_scene_ocr/models/test_100/'
    image_path = '/home/share/model_share/box_5000_3900_down/4/'
    
    starttime = datetime.datetime.now()
    viz = False
    
    print 'start time is ', starttime
    
    down_cnt = 0
    down_rgt = 0
    for i in os.listdir(image_path):
        if '.txt' in i:
            continue
        img = cv2.imread(image_path+i)
        img = img[:, :, ::-1]
        #factor = min(img.shape[0]/step, img.shape[1]/step)
        #factor = 1.0
        #if img.shape[0] * img.shape[1] >= 2048**2:
            #factor = 1.0
        #else:
            #factor = 0.5
        if img.shape[0] * img.shape[1] >= 2048**2:
            factor = 2.0
        else:
            factor = 1.0
        reshape = (int(img.shape[1]/factor), int(img.shape[0]/factor))
        img = cv2.resize(img, reshape)
        if img.shape[0] < 227 or img.shape[1] < 227:
            continue
        result = model.predict(np.array([img]))
        if viz:
            figure_result(img, result[0])
        result = predict(result)
        down_cnt += 1
        if result == 3:
            down_rgt += 1
        else:
            cv2.imwrite('results/down/%s_%d_%d.jpg'%(i[:-4], down_cnt, result), img[:, :, ::-1])
    
    right_cnt = 0
    right_rgt = 0
    for i in os.listdir(image_path):
        if '.txt' in i:
            continue
        img = cv2.imread(image_path+i)
        img = np.transpose(img, [1, 0, 2])[::-1]
        img = img[:, :, ::-1]
        #factor = min(img.shape[0]/step, img.shape[1]/step)
        #factor = 1.0
        #if img.shape[0] * img.shape[1] >= 2048**2:
            #factor = 1.0
        #else:
            #factor = 0.5
        if img.shape[0] * img.shape[1] >= 2048**2:
            factor = 2.0
        else:
            factor = 1.0
        reshape = (int(img.shape[1]/factor), int(img.shape[0]/factor))
        img = cv2.resize(img, reshape)
        if img.shape[0] < 227 or img.shape[1] < 227:
            continue
        result = model.predict(np.array([img]))
        if viz:
            figure_result(img, result[0])
        result = predict(result)
        right_cnt += 1
        if result == 2:
            right_rgt += 1
        else:
            cv2.imwrite('results/right/%s_%d_%d.jpg'%(i[:-4], right_cnt, result), img[:, :, ::-1])
    
    up_cnt = 0
    up_rgt = 0
    for i in os.listdir(image_path):
        if '.txt' in i:
            continue
        img = cv2.imread(image_path+i)
        img = img[::-1, ::-1]
        img = img[:, :, ::-1]
        #factor = min(img.shape[0]/step, img.shape[1]/step)
        #factor = 1.0
        #if img.shape[0] * img.shape[1] >= 2048**2:
            #factor = 1.0
        #else:
            #factor = 0.5
        if img.shape[0] * img.shape[1] >= 2048**2:
            factor = 2.0
        else:
            factor = 1.0
        reshape = (int(img.shape[1]/factor), int(img.shape[0]/factor))
        img = cv2.resize(img, reshape)
        if img.shape[0] < 227 or img.shape[1] < 227:
            continue
        result = model.predict(np.array([img]))
        if viz:
            figure_result(img, result[0])
        result = predict(result)
        up_cnt += 1
        if result == 1:
            up_rgt += 1
        else:
            cv2.imwrite('results/up/%s_%d_%d.jpg'%(i[:-4], up_cnt, result), img[:, :, ::-1])
    
    left_cnt = 0
    left_rgt = 0
    for i in os.listdir(image_path):
        if '.txt' in i:
            continue
        img = cv2.imread(image_path+i)
        img = np.transpose(img, [1, 0, 2])[:, ::-1]
        img = img[:, :, ::-1]
        #factor = min(img.shape[0]/step, img.shape[1]/step)
        #factor = 1.0
        #if img.shape[0] * img.shape[1] >= 2048**2:
            #factor = 1.0
        #else:
            #factor = 0.5
        if img.shape[0] * img.shape[1] >= 2048**2:
            factor = 2.0
        else:
            factor = 1.0
        reshape = (int(img.shape[1]/factor), int(img.shape[0]/factor))
        img = cv2.resize(img, reshape)
        if img.shape[0] < 227 or img.shape[1] < 227:
            continue
        result = model.predict(np.array([img]))
        if viz:
            figure_result(img, result[0])
        result = predict(result)
        left_cnt += 1
        if result == 4:
            left_rgt += 1
        else:
            cv2.imwrite('results/left/%s_%d_%d.jpg'%(i[:-4], left_cnt, result), img[:, :, ::-1])
        
    print 'the downside rate is %.3f'%(float(down_rgt)/down_cnt)
    print 'the right side rate is %.3f'%(float(right_rgt)/right_cnt)
    print 'the upside rate is %.3f'%(float(up_rgt)/up_cnt)
    print 'the left side rate is %.3f'%(float(left_rgt)/left_cnt)
    print 'the total rate is %.3f'%(float(down_rgt+right_rgt+up_rgt+left_rgt)/
                                    (down_cnt+right_cnt+up_cnt+left_cnt))

    endtime = datetime.datetime.now()
    print 'the total time is ', endtime - starttime
    
    print 'end'
    
    

if __name__ == '__main__':
    main()
