import numpy as np
import glob
import os
import tensorflow as tf
import scipy.io
import math     
import random
import itertools
import PIL
import time
import datetime

from tensorflow.python.keras.callbacks import TensorBoard, LearningRateScheduler
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Lambda, AveragePooling2D
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.optimizers import SGD, Adam
from scipy.ndimage.filters import gaussian_filter 
from scipy.linalg import block_diag
from tensorflow.python.keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.preprocessing import image
from datagen import DataGenerator
from tensorflow.python.keras import regularizers

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
#    yPred = tf.Print(yPred, [yPred], message='\nRanking_yPred = ', summarize=25)
    M_tensor = K.constant(M)
    differences = K.dot(M_tensor,yPred)
    zeros_tensor = K.zeros(shape=(50, 1))    
    max_tensor = K.maximum(differences,zeros_tensor)    
    ranking_loss = K.sum(max_tensor)
    return ranking_loss  

def euclideanDistanceCountingLoss(yTrue,yPred):
#    yPred = tf.Print(yPred, [yPred], message='\nCounting_yPred = ', summarize=4900)
    subtraction = yTrue - yPred
    sq = K.square(subtraction)
    counting_loss = K.mean(sq, axis=None, keepdims=False)
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
    
# learning rate schedule
def step_decay(epoch):
    initial_lrate = 1e-6
    drop = 0.1
    epochs_drop = int(round((iterations/iterations_per_epoch)/2))
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
    
def downsample(im, size):
    rng_y = np.linspace(0, im.shape[0], size[0]+1).astype('int')
    rng_y = list(zip(rng_y[:-1], rng_y[1:]))
    rng_x = np.linspace(0, im.shape[1], size[1]+1).astype('int')
    rng_x = list(zip(rng_x[:-1], rng_x[1:]))        
    res = np.zeros(size)
    for (yi, yr) in enumerate(rng_y):
        for (xi, xr) in enumerate(rng_x):
            res[yi, xi] = im[yr[0]:yr[1], xr[0]:xr[1]].sum()    
    return res
    
from os.path import isfile
from pickle import dump, load
def pickle(fname, obj):
    with open(fname, 'wb') as fd:
        dump(obj, fd)

def unpickle(fname):
    with open(fname, 'rb') as fd:
        r = load(fd)
    return r

def multiscale_pyramid(images, labels, start=0.7, end=1.1):
    if isfile('images_pyramids.pkl'):
        print('Loading cached scale pyramid...')
        images_pyramids = unpickle('images_pyramids.pkl')
        labels_pyramids = unpickle('labels_pyramids.pkl')

    else:
        print('Creating scale pyramid...')
        interval = np.linspace(start, end, pyramid_scales)
        images_pyramids = {}
        labels_pyramids = {}
        for i, imgpath in enumerate(images):
            print(i,imgpath)
            img = image.load_img(imgpath)
            pyramid_images = []
            pyramid_labels = []
            for scale in interval:
                w, h = img.size
                
                new_w = int(round(w*math.sqrt(scale)))
                new_h = int(round(h*math.sqrt(scale)))
                resized_img = img.resize((new_w,new_h),PIL.Image.BILINEAR)
                pyramid_images.append(resized_img)
            
                resized_gt = downsample(labels[imgpath], (new_h,new_w))
                pyramid_labels.append(resized_gt)
            images_pyramids[imgpath] = pyramid_images
            labels_pyramids[imgpath] = pyramid_labels
        pickle('images_pyramids.pkl', images_pyramids)
        pickle('labels_pyramids.pkl', labels_pyramids)
    return (images_pyramids, labels_pyramids)

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
    counting_dataset_pyramid, train_labels_pyramid = multiscale_pyramid(counting_dataset, train_labels)
    
    # Ranking Dataset  
    ranking_dataset_path = 'ranking_data'  
    ranking_dataset = list()
    for im_path in glob.glob(os.path.join(ranking_dataset_path, '*.jpg')):
        ranking_dataset.append(im_path) 

    split_val_labels = {}        
    
    mae_sum = 0.0
    mse_sum = 0.0

    # create folder to save results
    date = str(datetime.datetime.now())
    d = date.split()
    d1 = d[0]
    d2 = d[1].split(':')    
    results_folder = 'Results-'+d1+'-'+d2[0]+'.'+d2[1]
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)    
    
    # 5-fold cross validation
    epochs = int(round(iterations/iterations_per_epoch))
    n_fold = 5
    for f in range(0,n_fold):
        print('\nFold '+str(f))
        
        # Model
        model = VGG16(include_top=False, weights='imagenet') 
        transfer_layer = model.get_layer('block5_conv3')
        conv_model = Model(inputs=[model.input], outputs=[transfer_layer.output],name='vgg_partial')
        
        counting_input = Input(shape=(224, 224, 3), dtype='float32', name='counting_input')
        ranking_input = Input(shape=(224, 224, 3), dtype='float32', name='ranking_input')
        x = conv_model([counting_input,ranking_input])
        counting_output = Conv2D(1, (3, 3),strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name='counting_output')(x)
        
        # The ranking output is computed using SUM pool. Here I use
        # GlobalAveragePooling2D followed by a multiplication by 14^2 to do
        # this.
        ranking_output = GlobalAveragePooling2D(name='ranking_output')(counting_output)
        train_model = Model(inputs=[counting_input,ranking_input], outputs=[counting_output,ranking_output])
        train_model.summary()
        
        # l2 weight decay
        for layer in train_model.layers:
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer = regularizers.l2(5e-4)        
            elif layer.name == 'vgg_partial':
                for l in layer.layers:                    
                    if hasattr(l, 'kernel_regularizer'):
                        l.kernel_regularizer = regularizers.l2(5e-4)                                  
                
        optimizer = SGD(lr=0.0, decay=0.0, momentum=0.9, nesterov=False)
#        optimizer = Adam(lr=0.0,decay=0.0)
        loss={'counting_output': euclideanDistanceCountingLoss, 'ranking_output': pairwiseRankingHingeLoss}
        loss_weights=[1.0, 0.00001]
        train_model.compile(optimizer=optimizer,
                        loss=loss,
                        loss_weights=loss_weights)                      
        
        if f == 0:
            split_train = ['counting_data_UCF/25.jpg','counting_data_UCF/49.jpg','counting_data_UCF/18.jpg','counting_data_UCF/13.jpg','counting_data_UCF/28.jpg','counting_data_UCF/34.jpg','counting_data_UCF/17.jpg','counting_data_UCF/3.jpg','counting_data_UCF/26.jpg','counting_data_UCF/15.jpg','counting_data_UCF/31.jpg','counting_data_UCF/6.jpg','counting_data_UCF/33.jpg','counting_data_UCF/2.jpg','counting_data_UCF/30.jpg','counting_data_UCF/36.jpg','counting_data_UCF/42.jpg','counting_data_UCF/20.jpg','counting_data_UCF/38.jpg','counting_data_UCF/11.jpg','counting_data_UCF/5.jpg','counting_data_UCF/7.jpg','counting_data_UCF/4.jpg','counting_data_UCF/21.jpg','counting_data_UCF/27.jpg','counting_data_UCF/39.jpg','counting_data_UCF/22.jpg','counting_data_UCF/43.jpg','counting_data_UCF/32.jpg','counting_data_UCF/35.jpg','counting_data_UCF/8.jpg','counting_data_UCF/50.jpg','counting_data_UCF/12.jpg','counting_data_UCF/19.jpg','counting_data_UCF/44.jpg','counting_data_UCF/23.jpg','counting_data_UCF/9.jpg','counting_data_UCF/46.jpg','counting_data_UCF/16.jpg','counting_data_UCF/41.jpg']
            split_val = ['counting_data_UCF/37.jpg','counting_data_UCF/48.jpg','counting_data_UCF/29.jpg','counting_data_UCF/10.jpg','counting_data_UCF/14.jpg','counting_data_UCF/1.jpg','counting_data_UCF/45.jpg','counting_data_UCF/47.jpg','counting_data_UCF/40.jpg','counting_data_UCF/24.jpg']
            split_val_labels = {k: val_labels[k] for k in split_val}
        
        elif f == 1:
            split_train = ['counting_data_UCF/37.jpg','counting_data_UCF/48.jpg','counting_data_UCF/29.jpg','counting_data_UCF/10.jpg','counting_data_UCF/14.jpg','counting_data_UCF/1.jpg','counting_data_UCF/45.jpg','counting_data_UCF/47.jpg','counting_data_UCF/40.jpg','counting_data_UCF/24.jpg','counting_data_UCF/31.jpg','counting_data_UCF/6.jpg','counting_data_UCF/33.jpg','counting_data_UCF/2.jpg','counting_data_UCF/30.jpg','counting_data_UCF/36.jpg','counting_data_UCF/42.jpg','counting_data_UCF/20.jpg','counting_data_UCF/38.jpg','counting_data_UCF/11.jpg','counting_data_UCF/5.jpg','counting_data_UCF/7.jpg','counting_data_UCF/4.jpg','counting_data_UCF/21.jpg','counting_data_UCF/27.jpg','counting_data_UCF/39.jpg','counting_data_UCF/22.jpg','counting_data_UCF/43.jpg','counting_data_UCF/32.jpg','counting_data_UCF/35.jpg','counting_data_UCF/8.jpg','counting_data_UCF/50.jpg','counting_data_UCF/12.jpg','counting_data_UCF/19.jpg','counting_data_UCF/44.jpg','counting_data_UCF/23.jpg','counting_data_UCF/9.jpg','counting_data_UCF/46.jpg','counting_data_UCF/16.jpg','counting_data_UCF/41.jpg']
            split_val = ['counting_data_UCF/25.jpg','counting_data_UCF/49.jpg','counting_data_UCF/18.jpg','counting_data_UCF/13.jpg','counting_data_UCF/28.jpg','counting_data_UCF/34.jpg','counting_data_UCF/17.jpg','counting_data_UCF/3.jpg','counting_data_UCF/26.jpg','counting_data_UCF/15.jpg']
            split_val_labels = {k: val_labels[k] for k in split_val}
        
        elif f == 2:
            split_train = ['counting_data_UCF/37.jpg','counting_data_UCF/48.jpg','counting_data_UCF/29.jpg','counting_data_UCF/10.jpg','counting_data_UCF/14.jpg','counting_data_UCF/1.jpg','counting_data_UCF/45.jpg','counting_data_UCF/47.jpg','counting_data_UCF/40.jpg','counting_data_UCF/24.jpg','counting_data_UCF/25.jpg','counting_data_UCF/49.jpg','counting_data_UCF/18.jpg','counting_data_UCF/13.jpg','counting_data_UCF/28.jpg','counting_data_UCF/34.jpg','counting_data_UCF/17.jpg','counting_data_UCF/3.jpg','counting_data_UCF/26.jpg','counting_data_UCF/15.jpg','counting_data_UCF/5.jpg','counting_data_UCF/7.jpg','counting_data_UCF/4.jpg','counting_data_UCF/21.jpg','counting_data_UCF/27.jpg','counting_data_UCF/39.jpg','counting_data_UCF/22.jpg','counting_data_UCF/43.jpg','counting_data_UCF/32.jpg','counting_data_UCF/35.jpg','counting_data_UCF/8.jpg','counting_data_UCF/50.jpg','counting_data_UCF/12.jpg','counting_data_UCF/19.jpg','counting_data_UCF/44.jpg','counting_data_UCF/23.jpg','counting_data_UCF/9.jpg','counting_data_UCF/46.jpg','counting_data_UCF/16.jpg','counting_data_UCF/41.jpg']
            split_val = ['counting_data_UCF/31.jpg','counting_data_UCF/6.jpg','counting_data_UCF/33.jpg','counting_data_UCF/2.jpg','counting_data_UCF/30.jpg','counting_data_UCF/36.jpg','counting_data_UCF/42.jpg','counting_data_UCF/20.jpg','counting_data_UCF/38.jpg','counting_data_UCF/11.jpg']
            split_val_labels = {k: val_labels[k] for k in split_val}
        
        elif f == 3:
            split_train = ['counting_data_UCF/37.jpg','counting_data_UCF/48.jpg','counting_data_UCF/29.jpg','counting_data_UCF/10.jpg','counting_data_UCF/14.jpg','counting_data_UCF/1.jpg','counting_data_UCF/45.jpg','counting_data_UCF/47.jpg','counting_data_UCF/40.jpg','counting_data_UCF/24.jpg','counting_data_UCF/25.jpg','counting_data_UCF/49.jpg','counting_data_UCF/18.jpg','counting_data_UCF/13.jpg','counting_data_UCF/28.jpg','counting_data_UCF/34.jpg','counting_data_UCF/17.jpg','counting_data_UCF/3.jpg','counting_data_UCF/26.jpg','counting_data_UCF/15.jpg','counting_data_UCF/31.jpg','counting_data_UCF/6.jpg','counting_data_UCF/33.jpg','counting_data_UCF/2.jpg','counting_data_UCF/30.jpg','counting_data_UCF/36.jpg','counting_data_UCF/42.jpg','counting_data_UCF/20.jpg','counting_data_UCF/38.jpg','counting_data_UCF/11.jpg','counting_data_UCF/8.jpg','counting_data_UCF/50.jpg','counting_data_UCF/12.jpg','counting_data_UCF/19.jpg','counting_data_UCF/44.jpg','counting_data_UCF/23.jpg','counting_data_UCF/9.jpg','counting_data_UCF/46.jpg','counting_data_UCF/16.jpg','counting_data_UCF/41.jpg']
            split_val = ['counting_data_UCF/5.jpg','counting_data_UCF/7.jpg','counting_data_UCF/4.jpg','counting_data_UCF/21.jpg','counting_data_UCF/27.jpg','counting_data_UCF/39.jpg','counting_data_UCF/22.jpg','counting_data_UCF/43.jpg','counting_data_UCF/32.jpg','counting_data_UCF/35.jpg']
            split_val_labels = {k: val_labels[k] for k in split_val}
        
        elif f == 4:
            split_train = ['counting_data_UCF/37.jpg','counting_data_UCF/48.jpg','counting_data_UCF/29.jpg','counting_data_UCF/10.jpg','counting_data_UCF/14.jpg','counting_data_UCF/1.jpg','counting_data_UCF/45.jpg','counting_data_UCF/47.jpg','counting_data_UCF/40.jpg','counting_data_UCF/24.jpg','counting_data_UCF/25.jpg','counting_data_UCF/49.jpg','counting_data_UCF/18.jpg','counting_data_UCF/13.jpg','counting_data_UCF/28.jpg','counting_data_UCF/34.jpg','counting_data_UCF/17.jpg','counting_data_UCF/3.jpg','counting_data_UCF/26.jpg','counting_data_UCF/15.jpg','counting_data_UCF/31.jpg','counting_data_UCF/6.jpg','counting_data_UCF/33.jpg','counting_data_UCF/2.jpg','counting_data_UCF/30.jpg','counting_data_UCF/36.jpg','counting_data_UCF/42.jpg','counting_data_UCF/20.jpg','counting_data_UCF/38.jpg','counting_data_UCF/11.jpg','counting_data_UCF/5.jpg','counting_data_UCF/7.jpg','counting_data_UCF/4.jpg','counting_data_UCF/21.jpg','counting_data_UCF/27.jpg','counting_data_UCF/39.jpg','counting_data_UCF/22.jpg','counting_data_UCF/43.jpg','counting_data_UCF/32.jpg','counting_data_UCF/35.jpg']
            split_val = ['counting_data_UCF/8.jpg','counting_data_UCF/50.jpg','counting_data_UCF/12.jpg','counting_data_UCF/19.jpg','counting_data_UCF/44.jpg','counting_data_UCF/23.jpg','counting_data_UCF/9.jpg','counting_data_UCF/46.jpg','counting_data_UCF/16.jpg','counting_data_UCF/41.jpg']
            split_val_labels = {k: val_labels[k] for k in split_val}
                
        counting_dataset_pyramid_split = []
        train_labels_pyramid_split = []
        for key in split_train:
            counting_dataset_pyramid_split.append(counting_dataset_pyramid[key][0])
            counting_dataset_pyramid_split.append(counting_dataset_pyramid[key][1])
            counting_dataset_pyramid_split.append(counting_dataset_pyramid[key][2])
            counting_dataset_pyramid_split.append(counting_dataset_pyramid[key][3])
            counting_dataset_pyramid_split.append(counting_dataset_pyramid[key][4])
            
            train_labels_pyramid_split.append(train_labels_pyramid[key][0])
            train_labels_pyramid_split.append(train_labels_pyramid[key][1])
            train_labels_pyramid_split.append(train_labels_pyramid[key][2])
            train_labels_pyramid_split.append(train_labels_pyramid[key][3])
            train_labels_pyramid_split.append(train_labels_pyramid[key][4])
            
        index_shuf = np.arange(len(counting_dataset_pyramid_split))
        np.random.shuffle(index_shuf)
        counting_dataset_pyramid_split_shuf = []
        train_labels_pyramid_split_shuf = []
        for i in index_shuf:
            counting_dataset_pyramid_split_shuf.append(counting_dataset_pyramid_split[i])
            train_labels_pyramid_split_shuf.append(train_labels_pyramid_split[i])                
        
        train_generator = DataGenerator(counting_dataset_pyramid_split_shuf, train_labels_pyramid_split_shuf, ranking_dataset, **params)
        lrate = LearningRateScheduler(step_decay)
        callbacks_list = [lrate]
        train_model.fit_generator(generator=train_generator, epochs=epochs, callbacks=callbacks_list)
                
        #test images original size       
        tmp_model = train_model.get_layer('vgg_partial')
        test_input = Input(shape=(None, None, 3), dtype='float32', name='test_input')
        new_input = tmp_model(test_input)
        co = train_model.get_layer('counting_output')(new_input)
        test_output = Lambda(lambda i: K.sum(i,axis=(1,2)), name='test_output')(co)
        test_model = Model(inputs=[test_input], outputs=[test_output])
        
        predictions = np.empty((len(split_val),1))
        y_validation = np.empty((len(split_val),1))
        for i in range(len(split_val)):
            img = image.load_img(split_val[i])
            img_to_array = image.img_to_array(img)
            img_to_array = preprocess_input(img_to_array)
            img_to_array = np.expand_dims(img_to_array, axis=0)
        
            pred_test = test_model.predict(img_to_array)
            predictions[i] = pred_test[0]
            y_validation[i] = split_val_labels[split_val[i]]
        
        mean_abs_err = mae(predictions, y_validation)
        mean_sqr_err = mse(predictions, y_validation)               
        
        print('\n######################')
        print('Results on TEST SPLIT:')
        print(' MAE: {}'.format(mean_abs_err))
        print(' MSE: {}'.format(mean_sqr_err))
        print("Took %f seconds" % (time.time() - s))
        path1 = results_folder+'/test_split_results_fold-'+str(f)+'.txt'
        with open(path1, 'w') as f:
            f.write('mae: %f,\nmse: %f, \nTook %f seconds' % (mean_abs_err,mean_sqr_err,time.time() - s))

        mae_sum = mae_sum + mean_abs_err
        mse_sum = mse_sum + mean_sqr_err
    
    print('\n################################')
    print('Average Results on TEST SPLIT:')    
    print(' AVE MAE: {}'.format(mae_sum/n_fold))
    print(' AVE MSE: {}'.format(mse_sum/n_fold))
    print("Took %f seconds" % (time.time() - s))
    path2 = results_folder+'/test_split_results_avg.txt'
    with open(path2, 'w') as f:
        f.write('avg_mae: %f, \navg_mse: %f, \nTook %f seconds' % (mae_sum/n_fold,mse_sum/n_fold,time.time() - s))
        
if __name__ == "__main__":
    s = time.time()
    batch_size = 25
    iterations = 20000
    pyramid_scales = 5
    train_split_length = 40
    iterations_per_epoch = int(round((train_split_length * pyramid_scales)/batch_size))
    M = createMatrixForLoss(batch_size)
    params = {'dim': (224,224),
              'batch_size': batch_size, 
              'n_channels': 3,
              'shuffle': True,
              'rank_images': int(round(batch_size/5))}
    main()