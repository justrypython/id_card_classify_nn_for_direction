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
    weights_path = 'model/convnet_direction_227_weights_epoch10_loss0.0524.h5'

    #-----------------------------------------------------------------
    # 2: Build the Keras model
    model = convnet('alexnet', weights_path=weights_path, heatmap=True)
    
    #image_path = '/home/share/model_share/ks_nature_scene_ocr/models/test_100/'
    image_path = 'results/badcase/'
    #image_path = 'testimg/angle_1/'
    
    starttime = datetime.datetime.now()
    viz = True
    
    print 'start time is ', starttime
    
    for i in os.listdir(image_path):
        if '.txt' in i:
            continue
        print(i)
        img = cv2.imread(image_path+i)
        img = img[:, :, ::-1]
        print('image shape : ', img.shape[:2])
        #factor = min(img.shape[0]/step, img.shape[1]/step)
        factor = 1.0
        for factor in [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0]:
            print(factor)
            reshape = (int(img.shape[1]/factor), int(img.shape[0]/factor))
            newimg = cv2.resize(img, reshape)
            print('new image shape ', newimg.shape[:2])
            if newimg.shape[0] < 227 or newimg.shape[1] < 227:
                continue
            if newimg.shape[0] * newimg.shape[1] > 4096**2:
                continue
            result = model.predict(np.array([newimg]))
            if viz:
                figure_result(newimg, result[0])
            result = predict(result)
            print(result)

    endtime = datetime.datetime.now()
    print 'the total time is ', endtime - starttime
    
    print 'end'
    
    

if __name__ == '__main__':
    main()
