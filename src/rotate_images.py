#encoding:UTF-8

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


path = '/media/zhaoke/b0685ee4-63e3-4691-ae02-feceacff6996/data/'
dst_path = '/media/zhaoke/b0685ee4-63e3-4691-ae02-feceacff6996/rctw17_east/'

def rotate_right(x):
    def _rotate_right(*args):
        return args[0][0], x-args[0][1]
    return _rotate_right

def rotate_left(y):
    def _rotate_left(*args):
        return y-args[0][0], args[0][1]
    return _rotate_left

def rotate_up(x, y):
    def _rotate_left(*args):
        return x - args[0][0], y-args[0][1]
    return _rotate_left
    

def main():
    paths = [i for i in os.listdir(path) if '.jpg' in i]
    #right
    for i in paths[:100]:
        img = cv2.imread(path+i)
        y, x = img.shape[:2]
        right_func = rotate_right(x)
        img = np.transpose(img, [1, 0, 2])[::-1]
        cv2.imwrite(dst_path+'right/'+i, img)
        f = open(path+i.replace('.jpg', '.txt'))
        lines = f.readlines()
        f.close()
        lines = [j.replace('\n', '').replace('\r', '').split(',') for j in lines]
        lines = np.array(lines).astype(np.int)
        polys = lines[:, :8]
        labels = lines[:, -1:]
        polys = polys.reshape([-1, 4, 2])
        polys = polys[:, :, ::-1]
        polys = np.apply_along_axis(right_func, -1, polys)
        polys = polys.reshape((-1, 8))
        polys = np.concatenate([polys, labels], axis=-1)
        polys = polys.astype(np.str)
        polys = '\n'.join([','.join(j) for j in polys])
        f = open(dst_path+'right/'+i.replace('.jpg', '.txt'), 'w')
        f.write(polys)
        f.close()
    #left
    for i in paths[:100]:
        img = cv2.imread(path+i)
        y, x = img.shape[:2]
        left_func = rotate_left(y)
        img = np.transpose(img, [1, 0, 2])[:, ::-1]
        cv2.imwrite(dst_path+'left/'+i, img)
        f = open(path+i.replace('.jpg', '.txt'))
        lines = f.readlines()
        f.close()
        lines = [j.replace('\n', '').replace('\r', '').split(',') for j in lines]
        lines = np.array(lines).astype(np.int)
        polys = lines[:, :8]
        labels = lines[:, -1:]
        polys = polys.reshape([-1, 4, 2])
        polys = polys[:, :, ::-1]
        polys = np.apply_along_axis(left_func, -1, polys)
        polys = polys.reshape((-1, 8))
        polys = np.concatenate([polys, labels], axis=-1)
        polys = polys.astype(np.str)
        polys = '\n'.join([','.join(j) for j in polys])
        f = open(dst_path+'left/'+i.replace('.jpg', '.txt'), 'w')
        f.write(polys)
        f.close()
    #up
    for i in paths[:100]:
        img = cv2.imread(path+i)
        y, x = img.shape[:2]
        up_func = rotate_up(x, y)
        img = img[::-1, ::-1]
        cv2.imwrite(dst_path+'up/'+i, img)
        f = open(path+i.replace('.jpg', '.txt'))
        lines = f.readlines()
        f.close()
        lines = [j.replace('\n', '').replace('\r', '').split(',') for j in lines]
        lines = np.array(lines).astype(np.int)
        polys = lines[:, :8]
        labels = lines[:, -1:]
        polys = polys.reshape([-1, 4, 2])
        polys = np.apply_along_axis(up_func, -1, polys)
        #polys = polys[:, ::-1]
        polys = polys.reshape((-1, 8))
        polys = np.concatenate([polys, labels], axis=-1)
        polys = polys.astype(np.str)
        polys = '\n'.join([','.join(j) for j in polys])
        f = open(dst_path+'up/'+i.replace('.jpg', '.txt'), 'w')
        f.write(polys)
        f.close()

if __name__ == '__main__':
    main()