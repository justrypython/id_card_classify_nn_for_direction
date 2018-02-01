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

def predict(result, thres=0.5):
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
    #weights_path = 'model/convnet_direction_227_weights_epoch03_loss0.0069.h5'
    weights_path = 'model/convnet_direction_227_weights_epoch04_loss0.0673.h5'

    #-----------------------------------------------------------------
    # 2: Build the Keras model
    model = convnet('alexnet', weights_path=weights_path, heatmap=True)
    
    #image_path = '/media/zhaoke/b0685ee4-63e3-4691-ae02-feceacff6996/data/'
    image_path = 'angle/'
    dst_path = 'results/'
    
    starttime = datetime.datetime.now()
    viz = False
    
    image_paths = []
    for root, dirs, files in os.walk(image_path):
        for i in files:
            image_paths.append(os.path.join(root, i))
    
    print 'start time is ', starttime
    
    cnt = 0
    for i in image_paths:
        if '.txt' in i:
            continue
        img = cv2.imread(i)
        img = img[:, :, ::-1]
        #factor = min(img.shape[0]/step, img.shape[1]/step)
        factor = 1.0
        reshape = (int(img.shape[1]/factor), int(img.shape[0]/factor))
        img = cv2.resize(img, reshape)
        if img.shape[0] < 227 or img.shape[1] < 227:
            continue
        result = model.predict(np.array([img]))
        if viz:
            figure_result(img, result[0])
        result = predict(result)
        if result == 0:
            txt = i.replace('.jpg', '.txt')
            filename = i[-i[::-1].find('/'):]
            os.system('cp %s %sbgd/%s'%(i, dst_path, filename))
            #os.system('cp %s %sbgd/%s'%(txt, dst_path, filename.replace('.jpg', '.txt')))
        elif result == 1:
            txt = i.replace('.jpg', '.txt')
            filename = i[-i[::-1].find('/'):]
            os.system('cp %s %sup/%s'%(i, dst_path, filename))
            #os.system('cp %s %sup/%s'%(txt, dst_path, filename.replace('.jpg', '.txt')))
        elif result == 2:
            txt = i.replace('.jpg', '.txt')
            filename = i[-i[::-1].find('/'):]
            os.system('cp %s %sright/%s'%(i, dst_path, filename))
            #os.system('cp %s %sright/%s'%(txt, dst_path, filename.replace('.jpg', '.txt')))
        elif result == 3:
            txt = i.replace('.jpg', '.txt')
            filename = i[-i[::-1].find('/'):]
            os.system('cp %s %sdown/%s'%(i, dst_path, filename))
            #os.system('cp %s %sdown/%s'%(txt, dst_path, filename.replace('.jpg', '.txt')))
        else:
            txt = i.replace('.jpg', '.txt')
            filename = i[-i[::-1].find('/'):]
            os.system('cp %s %sleft/%s'%(i, dst_path, filename))
            #os.system('cp %s %sleft/%s'%(txt, dst_path, filename.replace('.jpg', '.txt')))
        cnt += 1
        print(cnt)
    
    print 'end'
    
    

if __name__ == '__main__':
    main()
