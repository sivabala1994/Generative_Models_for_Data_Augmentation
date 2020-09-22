#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:11:27 2019

@author: siv.bala
"""

import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import numpy as np
from keras.models import model_from_json
from tf_explain.core.grad_cam import GradCAM

from tf_explain.core.smoothgrad import SmoothGrad
def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    elif epoch > 100:
        lrate = 0.0003        
    return lrate

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
#
##z-score
#mean = np.mean(x_train,axis=(0,1,2,3))
#std = np.std(x_train,axis=(0,1,2,3))
#x_train = (x_train-mean)/(std+1e-7)
#x_test = (x_test-mean)/(std+1e-7)
#
num_classes = 10
#y_train = np_utils.to_categorical(y_train,num_classes)
#y_test = np_utils.to_categorical(y_test,num_classes)

weight_decay = 1e-4
model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', kernel_initializer='glorot_uniform',kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(num_classes, activation='softmax',name='output'))

model.summary()




#save to disk

def savemodel(model_path,model):
    model_json = model.to_json()
    with open(model_path+'model.json', 'w') as json_file:
        json_file.write(model_json)
    
    model.save_weights(model_path+'model.h5')    


def loadmodel(model_path):
    json_file = open(model_path+"model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    
#    loaded_model.load_weights(model_path+"model.h5")
#    print("Loaded model from disk")
    return loaded_model

loaded_model=loadmodel('/Users/siv.bala/Desktop/keras_data/scripts/')

#training
#batch_size = 64

#opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
#loaded_model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
#loaded_model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),\
#                    steps_per_epoch=x_train.shape[0] // batch_size,epochs=125,\
#                    verbose=1,validation_data=(x_test,y_test),callbacks=[LearningRateScheduler(lr_schedule)])


config_param = {
   'num_epochs':1,
   'batch_size':128,
   'optimizer':'Adam',
   'learning_rate':0.01,
   'keep_checkpoint_max':5,
   'save_checkpoints_steps':50,
   'save_summary_steps':5,
   'throttle':1,
   'checkpoint_dir':'/Users/sathya/work/mldashboard/ezeeai/ezeeai/user_data/test/models/new_model/checkpoints',
   'custom_model':'/Users/sathya/work/mldashboard/ezeeai/ezeeai/user_data/test/models/new_model/custom',
   'export_dir':'/Users/sathya/work/mldashboard/ezeeai/ezeeai/user_data/test/models/new_model/checkpoints/export/best_exporter',
   'log_dir':'/Users/sathya/work/mldashboard/ezeeai/ezeeai/user_data/test/models/new_model/log',
   'tmp_dir':'/Users/sathya/work/mldashboard/ezeeai/ezeeai/user_data/test/models/new_model/tmp',
   'data_path':'/Users/sathya/work/mldashboard/ezeeai/ezeeai/user_data/test/models/new_model/adult.pkl',
   'email':'test@test.com',
   'custom_path':'/Users/sathya/work/mldashboard/ezeeai/ezeeai/user_data/test/models/new_model/custom',
   'transform_path':'/Users/sathya/work/mldashboard/ezeeai/ezeeai/user_data/test/models/new_model/transform',
   'loss_function':'categorical_crossentropy'
}






def modelfit(model,config_param,data_dir,test_dir):
    opt_rms = keras.optimizers.Adam(lr=config_param['learning_rate'],decay=1e-6)
    model.compile(loss=config_param['loss_function'], optimizer=opt_rms, metrics=['accuracy'])
    

    model.fit_generator(generate_data(data_dir,config_param['batch_size']),\
                    steps_per_epoch=x_train.shape[0] // config_param['batch_size'],epochs=config_param['num_epochs'],\
                    verbose=1,validation_data=test_data(test_dir),callbacks=[LearningRateScheduler(lr_schedule)])


data_dir="/Users/siv.bala/Desktop/keras_data/datasets/cifar10/train/labels.txt"
test_dir="/Users/siv.bala/Desktop/keras_data/datasets/cifar10/test/labels.txt"
explainer = SmoothGrad()


#modelfit(loaded_model,config_param,data_dir,test_dir)
grid = explainer.explain((x_train[1,:,:,:],1), model,'output', 1)
explainer.save(grid, '.', 'smoothgrad.png')
