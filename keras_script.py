import numpy as np
import glob
import os
import tensorflow as tf
import cv2
import scipy.io
import scipy.signal
import math     
import random
import itertools
import matplotlib.pyplot as plt

from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Conv2D, AveragePooling2D, GlobalAveragePooling2D, Lambda
from tensorflow.python.keras.applications import VGG16
from datagen import DataGenerator
from tensorflow.python.keras.optimizers import SGD, Adam
from scipy.ndimage.filters import gaussian_filter 
from scipy.linalg import block_diag
from tensorflow.python.keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.preprocessing import image

tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

def load_gt_from_mat(gt_file, gt_shape):
    gt = np.zeros(gt_shape, dtype='float32')
    mat_contents = scipy.io.loadmat(gt_file)
    oct_struct = mat_contents['annPoints']
    for dot in oct_struct:
        gt[int(math.floor(dot[1]))-1, int(math.floor(dot[0]))-1] = 1.0
    sigma = 15
    density_map = gaussian_filter(gt, sigma)
    return density_map, len(oct_struct)
    

def pairwiseRankingHingeLoss(yTrue,yPred):
    kvar = K.constant(M)
#    yPred = K.squeeze(yPred, 3)
#    yPred = K.squeeze(yPred, 2)
    
    differences = K.dot(kvar,yPred)# (50x25)*(25x1)=(50x1)    
    # differences = tf.Print(differences, [differences], message='\ndifferences = ', summarize=50)#50x1
    
    var = K.zeros(shape=(50, 1))    
    
    max_tensor = K.maximum(differences,var)
    # max_tensor = tf.Print(max_tensor, [max_tensor], message='\nmax_tensor = ', summarize=50)#50x1  
    
    ranking_loss = K.sum(max_tensor)
    # ranking_loss = tf.Print(ranking_loss, [ranking_loss], message='\nranking_loss = ', summarize=1)#1x1    

    return ranking_loss    
    
batch_size = 25 #25 for counting and 25 for ranking

Block_matrix = np.zeros((10,5)) # block to create the matrix M
for i in range(0,4):
    for j in range(0,5):
        if j == 0:
            Block_matrix[i][j] = -1
        if j == i+1:
            Block_matrix[i][j] = 1 

Block_matrix[4:7,1:5] = Block_matrix[0:3,0:4]          
Block_matrix[7:9,2:5] = Block_matrix[0:2,0:3]        
Block_matrix[9:10,3:5] = Block_matrix[0:1,0:2] 

blocks_number = int(round(batch_size/5)) # blocks number to create the matrix M
blocks_list = list()
for i in range(0,blocks_number):
    blocks_list.append(Block_matrix)
M = block_diag(*blocks_list) # matrix used in pairwiseRankingHingeLoss to compare sub-patches
# np.savetxt('Newfolder\\M_matrix.txt', M, fmt='%d')

# Counting Dataset
counting_dataset = list()

counting_dataset_path = 'counting_data_UCF'
for im in glob.glob(os.path.join(counting_dataset_path, '*.jpg')):
    counting_dataset.append(im)

# Counting ground truth    
train_labels = {}
val_labels = {}
for i in range(len(counting_dataset)):
    img = image.load_img(counting_dataset[i])
    gt_file = counting_dataset[i].replace('.jpg','_ann.mat')
    h,w = img.size
    dmap,crowd_number = load_gt_from_mat(gt_file, (w,h))
    train_labels[counting_dataset[i]] = dmap
    val_labels[counting_dataset[i]] = crowd_number

#Ranking Dataset    
ranking_dataset = list()

ranking_dataset_path = 'ranking_data'
for im in glob.glob(os.path.join(ranking_dataset_path, '*.jpg')):
    ranking_dataset.append(im)

# Design model
model = VGG16(include_top=True, weights='imagenet') #include_top=False to get only the convolutional part of VGG16
transfer_layer = model.get_layer('block5_conv3')
conv_model = Model(inputs=[model.input], outputs=[transfer_layer.output])

counting_input = Input(shape=(224, 224, 3), dtype='float32', name='counting_input')
ranking_input = Input(shape=(224, 224, 3), dtype='float32', name='ranking_input')
x = conv_model([counting_input,ranking_input])
#a single convolutional layer (a single 3x3x512 filter with stride 1 and zero padding to maintain same size)
counting_output = Conv2D(1, (3, 3),strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name='counting_output')(x)

# The ranking output is computed using SUM pool. Here I use
# GlobalAveragePooling2D followed by a multiplication by 14^2 to do
# this.
ranking_output = Lambda(lambda i: 14.0 * 14.0 * i, name='ranking_output')(GlobalAveragePooling2D()(counting_output))
new_model = Model(inputs=[counting_input,ranking_input], outputs=[counting_output,ranking_output])
new_model.summary()

optimizer = Adam(lr=1e-5)
loss={'counting_output': 'mean_squared_error', 'ranking_output': pairwiseRankingHingeLoss}
loss_weights=[1.0, 0.0]

new_model.compile(optimizer=optimizer,
                  loss=loss,
                  loss_weights=loss_weights)

print('##################FIT GENERATOR######################################################################')
# Parameters
params = {'dim': (224,224),
          'batch_size': batch_size, 
          'n_channels': 3,
          'shuffle': True,
          'rank_images': int(round(batch_size/5))} # numero di immagini di ranking da prendere per creare un batch (da ogni immagine vengono prese 5 sottopatch)
          
split_train_labels = {}
split_val_labels = {}

#RANDOMIZE the order of images before splitting
np.random.shuffle(counting_dataset)

split_size = int(round(len(counting_dataset)/5))
splits_list = list()
for t in range(5):
    splits_list.append(counting_dataset[t*split_size:t*split_size+split_size])    

###########################
# ADB: Evaluation functions
def mse(pred, gt):
    return np.sqrt(((pred - gt) ** 2.0).mean())

def mae(pred, gt):
    return abs(pred - gt).mean()

# 5-fold cross validation
epochs = 15
for f in range(0,1):
    print('Folder '+str(f))

    splits_list_tmp = splits_list.copy()
    
    #counting validation split
    split_val = splits_list_tmp[f]
    
    del splits_list_tmp[f]
    flat=itertools.chain.from_iterable(splits_list_tmp)
    
    #counting train split
    split_train = list(flat)

    #counting validation split labels
    split_val_labels = {k: val_labels[k] for k in split_val}    

    # counting train split labels
    split_train_labels = {k: train_labels[k] for k in split_train}
    
    # train for FIVE epochs.
    train_generator = DataGenerator(split_train, split_train_labels, ranking_dataset, **params)
    new_model.fit_generator(generator=train_generator, epochs=epochs)

    # Look at some outputs... Note, this is NOT how the model should
    # be evaluated. This is CROPPING and passing ranking images, but
    # rather you should pass ENTIRE test images and compare the
    # ranking_output with the ground truth COUNTS for each image.
    # test_generator = DataGenerator(split_val, split_val_labels, ranking_dataset, **params)
    # res = new_model.predict_generator(test_generator)
    # print('Count stats for split={}:'.format(f))
    # print(' min count: {}'.format(res[1].min()))
    # print(' max count: {}'.format(res[1].max()))
    # print(' avg count: {}'.format(res[1].mean()))
    # print(new_model.metrics_names)

    X_validation = np.empty((len(split_val), 224, 224, 3))
    y_validation = np.empty((len(split_val),1))
    for i in range(len(split_val)):   
        img = image.load_img(split_val[i], target_size=(224, 224))
        img_to_array = image.img_to_array(img)
        img_to_array = preprocess_input(img_to_array)
        X_validation[i,] = img_to_array
        y_validation[i] = split_val_labels[split_val[i]]
    
    # ADB: use model.predict() to get outputs, use own code for evaluation.
    pred_test = new_model.predict([X_validation, np.zeros((10, 224, 224, 3))])
    print('\n######################')
    print('Results on TEST SPLIT:')
    print(' MAE: {}'.format(mae(pred_test[1], y_validation)))
    print(' MSE: {}'.format(mse(pred_test[1], y_validation)))

    print('\n################################')
    tr_X = train_generator[0][0]['counting_input']
    tr_y = train_generator[0][1]['counting_output'].sum(1).sum(1).sum(1)
    pred_train = new_model.predict([tr_X, np.zeros((25, 224, 224, 3))])
    print('Results on FIRST TRAINING BATCH:')
    print(' MAE: {}'.format(mae(pred_test[1], y_validation)))
    print(' MSE: {}'.format(mse(pred_test[1], y_validation)))
