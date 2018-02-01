#encoding:UTF-8

import os
import numpy as np
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from convnets import convnet
import icdar


def main():
    #-----------------------------------------------------------------
    # 1: Set some necessary parameters
    data_path = 'model/tf_alexnet_direction_weights.h5'
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
    sgd = SGD(lr=0.01, decay=5e-4, momentum=0.9, nesterov=True)
    
    model = convnet('alexnet', weights_path=data_path, heatmap=False)
    model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])

    #-----------------------------------------------------------------
    # 4: Instantiate an encoder that can encode ground truth labels into 
    #    the format needed by the EAST loss function
    
    #-----------------------------------------------------------------
    # 5: Create the validation set batch generator
    data_generator = icdar.get_batch(num_workers=1,
                                     input_size=size,
                                     batch_size=1,
                                     labels=labels)
    valid_generator = icdar.get_batch(num_workers=1,
                                      input_size=size,
                                      batch_size=1,
                                      labels=labels)
    data_generator.next()
    
    #-----------------------------------------------------------------
    # 6: Run training
    model.fit_generator(generator = data_generator,
                        steps_per_epoch = 5000,
                        epochs = 100,
                        callbacks = [ModelCheckpoint('./model/convnet_direction_227_weights_epoch{epoch:02d}_loss{loss:.4f}.h5',
                                                     monitor='val_loss',
                                                     verbose=1,
                                                     save_best_only=True,
                                                     save_weights_only=True,
                                                     mode='auto',
                                                     period=1),
                                     ReduceLROnPlateau(monitor='val_loss',
                                                       factor=0.5,
                                                       patience=0,
                                                       epsilon=0.001,
                                                       cooldown=0)],
                        validation_data = valid_generator,
                        validation_steps = 500)
    
        

if __name__ == '__main__':
    main()
