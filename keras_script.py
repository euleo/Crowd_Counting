import numpy as np
import glob
import os
import tensorflow as tf
import scipy.io
import math     
import random
import itertools

from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Lambda
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.optimizers import SGD, Adam
from scipy.ndimage.filters import gaussian_filter 
from scipy.linalg import block_diag
from tensorflow.python.keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.preprocessing import image
from datagen import DataGenerator

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

def load_gt_from_mat(gt_file, gt_shape):
    '''
    @brief: This function creates density map from matlab file with points annotations.
    @param: gt_file: matlab file with annotated points.
    @param: gt_shape: density map shape.
    @return: density map and number of points for the input matlab file.
    '''
    
    gt = np.zeros(gt_shape, dtype='float32')
    mat_contents = scipy.io.loadmat(gt_file)
    dots = mat_contents['annPoints']
    for dot in dots:
        gt[int(math.floor(dot[1]))-1, int(math.floor(dot[0]))-1] = 1.0
    sigma = 15
    density_map = gaussian_filter(gt, sigma)
    return density_map, len(dots)

def pairwiseRankingHingeLoss(yTrue,yPred):
    '''
        @brief: This function gets net predictions for ranking output and calculate the pairwise ranking hinge loss.
        @param yTrue: dummy tensor, not used.
        @param yPred: net predictions for ranking output.
        @return: pairwise ranking hinge loss for ranking output.
    '''
    
    M_tensor = K.constant(M)
    differences = K.dot(M_tensor,yPred)    
    zeros_tensor = K.zeros(shape=(50, 1))    
    max_tensor = K.maximum(differences,zeros_tensor)    
    ranking_loss = K.sum(max_tensor)
    return ranking_loss  

def euclideanDistanceCountingLoss(yTrue,yPred):   
    counting_loss = K.mean(K.square(yPred - yTrue), axis=None, keepdims=False)
    return counting_loss    
    
def createMatrixForLoss(batch_size):
    '''
        @brief: This function creates the matrix used in pairwiseRankingHingeLoss to compare ranking sub-patches.
        @param batch_size: number of samples for a batch.
        @return: matrix to compare ranking sub-patches.
    '''
    
    block = np.zeros((10,5)) 
    for i in range(0,4):
        for j in range(0,5):
            if j == 0:
                block[i][j] = -1
            if j == i+1:
                block[i][j] = 1 

    block[4:7,1:5] = block[0:3,0:4]          
    block[7:9,2:5] = block[0:2,0:3]        
    block[9:10,3:5] = block[0:1,0:2] 

    blocks_number = int(round(batch_size/5)) 
    blocks_list = list()
    for i in range(0,blocks_number):
        blocks_list.append(block)
    return block_diag(*blocks_list) 
    
# ADB: Evaluation functions
def mse(pred, gt):
    return np.sqrt(((pred - gt) ** 2.0).mean())

def mae(pred, gt):
    return abs(pred - gt).mean()

def main():
    # Counting Dataset
    counting_dataset_path = 'counting_data_UCF'
    counting_dataset = list()
    train_labels = {}
    val_labels = {}
    for im_path in glob.glob(os.path.join(counting_dataset_path, '*.jpg')):
        counting_dataset.append(im_path)
        img = image.load_img(im_path)
        gt_file = im_path.replace('.jpg','_ann.mat')
        h,w = img.size
        dmap,crowd_number = load_gt_from_mat(gt_file, (w,h))
        train_labels[im_path] = dmap
        val_labels[im_path] = crowd_number

    # Ranking Dataset  
    ranking_dataset_path = 'ranking_data'  
    ranking_dataset = list()
    for im_path in glob.glob(os.path.join(ranking_dataset_path, '*.jpg')):
        ranking_dataset.append(im_path)

    # randomize the order of images before splitting
    np.random.shuffle(counting_dataset)

    split_size = int(round(len(counting_dataset)/5))
    splits_list = list()
    for t in range(5):
        splits_list.append(counting_dataset[t*split_size:t*split_size+split_size])    

    split_train_labels = {}
    split_val_labels = {}
    
    mae_sum = 0.0
    mse_sum = 0.0
    
    # 5-fold cross validation
    epochs = 1
    n_fold = 1
    for f in range(0,n_fold):
        print('\nFold '+str(f))
        
        # Model
        model = VGG16(include_top=True, weights='imagenet') 
        # model.summary()
        transfer_layer = model.get_layer('block5_conv3')
        conv_model = Model(inputs=[model.input], outputs=[transfer_layer.output])
        # conv_model.summary()
        
        counting_input = Input(shape=(224, 224, 3), dtype='float32', name='counting_input')
        ranking_input = Input(shape=(224, 224, 3), dtype='float32', name='ranking_input')
        x = conv_model([counting_input,ranking_input])
        counting_output = Conv2D(1, (3, 3),strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name='counting_output')(x)
        
        # The ranking output is computed using SUM pool. Here I use
        # GlobalAveragePooling2D followed by a multiplication by 14^2 to do
        # this.
        ranking_output = Lambda(lambda i: 14.0 * 14.0 * i, name='ranking_output')(GlobalAveragePooling2D()(counting_output))
        new_model = Model(inputs=[counting_input,ranking_input], outputs=[counting_output,ranking_output])
        # new_model.summary()
        
        optimizer = Adam(lr=1e-5)
        loss={'counting_output': euclideanDistanceCountingLoss, 'ranking_output': pairwiseRankingHingeLoss}
        loss_weights=[1.0, 0.0]
        
        new_model.compile(optimizer=optimizer,
                        loss=loss,
                        loss_weights=loss_weights)                      

        splits_list_tmp = splits_list.copy()
        
        # counting validation split
        split_val = splits_list_tmp[f]
        
        del splits_list_tmp[f]
        flat=itertools.chain.from_iterable(splits_list_tmp)
        
        # counting train split
        split_train = list(flat)

        # counting validation split labels
        split_val_labels = {k: val_labels[k] for k in split_val}    

        # counting train split labels
        split_train_labels = {k: train_labels[k] for k in split_train}        
        
        # train for FIVE epochs.
        train_generator = DataGenerator(split_train, split_train_labels, ranking_dataset[0:5], **params)
        new_model.fit_generator(generator=train_generator, epochs=epochs)

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
        mean_abs_err = mae(pred_test[1], y_validation)
        mean_sqr_err = mse(pred_test[1], y_validation)
        print('\n######################')
        print('Results on TEST SPLIT:')
        print(' MAE: {}'.format(mean_abs_err))
        print(' MSE: {}'.format(mean_sqr_err))

        mae_sum = mae_sum + mean_abs_err
        mse_sum = mse_sum + mean_sqr_err
        
        print('\n################################')
        tr_X = train_generator[0][0]['counting_input']
        tr_y = train_generator[0][1]['counting_output'].sum(1).sum(1).sum(1)
        pred_train = new_model.predict([tr_X, np.zeros((25, 224, 224, 3))])
        print('Results on FIRST TRAINING BATCH:')
        print(' MAE: {}'.format(mae(pred_train[1], tr_y)))
        print(' MSE: {}'.format(mse(pred_train[1], tr_y)))
    
    print('\n################################')
    print('Average Results on TEST SPLIT:')    
    print(' AVE MAE: {}'.format(mae_sum/n_fold))
    print(' AVE MSE: {}'.format(mse_sum/n_fold))
        
if __name__ == "__main__":
    batch_size = 25
    M = createMatrixForLoss(batch_size)    
    params = {'dim': (224,224),
              'batch_size': batch_size, 
              'n_channels': 3,
              'shuffle': True,
              'rank_images': int(round(batch_size/5))}
    main()