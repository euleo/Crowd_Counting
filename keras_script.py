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

from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Conv2D, AveragePooling2D
from tensorflow.python.keras.applications import VGG16
from datagen import DataGenerator
from tensorflow.python.keras.optimizers import SGD
from scipy.ndimage.filters import gaussian_filter 
from scipy.linalg import block_diag

tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

def load_gt_from_mat(gt_file, gt_shape):
    gt = np.zeros(gt_shape, dtype='float32')
    mat_contents = scipy.io.loadmat(gt_file)
    oct_struct = mat_contents['annPoints']
    for dot in oct_struct:
        gt[int(math.floor(dot[1]))-1,int(math.floor(dot[0]))-1] = 1.0
    sigma = 15
    gt = gt[:, :, 0]
    density_map = gaussian_filter(gt, sigma)
    return density_map
    
def pairwiseRankingHingeLoss(yTrue,yPred):    
    kvar = K.constant(M)

    yPred = K.squeeze(yPred, 3)
    yPred = K.squeeze(yPred, 2)
    
    differences = K.dot(kvar,yPred)
    
    var = K.zeros(shape=(50, 1))
    
    max_tensor = K.maximum(differences,var)
    
    ranking_loss = K.sum(max_tensor)
    return ranking_loss    
    
batch_size = 25 #25 for counting and 25 for ranking

#matrice usata nella pairwiseRankingHingeLoss per fare i confronti tra le 5 sottopatch di una immagine di ranking
Block_matrix = np.zeros((10,5))
for i in range(0,4):
    for j in range(0,5):
        if j == 0:
            Block_matrix[i][j] = -1
        if j == i+1:
            Block_matrix[i][j] = 1 

Block_matrix[4:7,1:5] = Block_matrix[0:3,0:4]          
Block_matrix[7:9,2:5] = Block_matrix[0:2,0:3]        
Block_matrix[9:10,3:5] = Block_matrix[0:1,0:2] 

#numero di Block_matrix che mi servono (per ogni 5 immagini di ranking mi serve una Block_matrix)
blocks_number = int(round(batch_size/5)) 
blocks_list = list()
for i in range(0,blocks_number):
    blocks_list.append(Block_matrix)
M = block_diag(*blocks_list) #matrice usata nella pairwiseRankingHingeLoss

# Counting Dataset
counting_dataset = list()

counting_dataset_path = 'counting_data_UCF'
for im in glob.glob(os.path.join(counting_dataset_path, '*.jpg')):
    counting_dataset.append(im)

# Counting ground truth    
labels = {}
for i in range(len(counting_dataset)):
    img = cv2.imread(counting_dataset[i])
    gt_file = counting_dataset[i].replace('.jpg','_ann.mat')
    dmap = load_gt_from_mat(gt_file, img.shape)
    labels[counting_dataset[i]] = dmap

#Ranking Dataset    
ranking_dataset = list()

ranking_dataset_path = 'ranking_data'
for im in glob.glob(os.path.join(ranking_dataset_path, '*.jpg')):
    ranking_dataset.append(im)

# # Design model
model = VGG16(include_top=True, weights='imagenet') #include_top=False to get only the convolutional part of VGG16
transfer_layer = model.get_layer('block5_conv3')
conv_model = Model(inputs=[model.input], outputs=[transfer_layer.output])

counting_input = Input(shape=(224, 224, 3), dtype='float32', name='counting_input')
ranking_input = Input(shape=(224, 224, 3), dtype='float32', name='ranking_input')
x = conv_model([counting_input,ranking_input])
#a single convolutional layer (a single 3x3x512 filter with stride 1 and zero padding to maintain same size)
counting_output = Conv2D(1, (3, 3),strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name='conv2d_1')(x)
ranking_output = AveragePooling2D(pool_size=(14, 14), strides=None, padding='valid', data_format=None, name='average_pooling2d_1')(counting_output)
new_model = Model(inputs=[counting_input,ranking_input], outputs=[counting_output,ranking_output])
new_model.summary()

optimizer = SGD(lr=0.000001, decay=0.0005, momentum=0.0, nesterov=False)
loss={'conv2d_1': 'mean_squared_error', 'average_pooling2d_1': pairwiseRankingHingeLoss}
loss_weights=[1,100]
metrics = {'average_pooling2d_1': ['mae', 'mse']}

new_model.compile(optimizer=optimizer,
              loss=loss,
			  metrics=metrics,
              loss_weights=loss_weights)

print('##################FIT GENERATOR######################################################################')
# Parameters
params = {'dim': (224,224),
          'batch_size': batch_size, 
          'n_channels': 3,
          'shuffle': False,
          'rank_images': int(round(batch_size/5))} # numero di immagini di ranking da prendere per creare un batch (da ogni immagine vengono prese 5 sottopatch)
          
split_train_labels = {}
split_val_labels = {}

split_size = int(round(len(counting_dataset)/5))
splits_list = list()
for t in range(0,5):
    splits_list.append(counting_dataset[t*split_size:t*split_size+split_size])    

# 5-fold cross validation
for f in range(0,1):
    print('Folder '+str(f))

    splits_list_tmp = splits_list.copy()
    
    #counting validation split
    split_val = splits_list_tmp[f]
    
    del splits_list_tmp[f]    
    flat=itertools.chain.from_iterable(splits_list_tmp)
    
    #counting train split
    split_train = list(flat)    
    
    #counting train split labels
    split_train_labels = {k: labels[k] for k in split_train}
    
    #counting validation split labels
    split_val_labels = {k: labels[k] for k in split_val}
        
    #ranking train split
    split_rank_train = ranking_dataset[0:50]
        
    #train
    # new_model.fit_generator(generator=DataGenerator(split_train, split_train_labels, split_rank_train, **params), epochs=1, callbacks=[tbCallBack])
    
    #train+validation
    new_model.fit_generator(generator=DataGenerator(split_train, split_train_labels, split_rank_train, **params), validation_data=DataGenerator(split_val, split_val_labels, split_rank_train, **params), epochs=1, callbacks=[tbCallBack])
