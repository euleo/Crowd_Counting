import numpy as np
import tensorflow as tf
import cv2
import random
import math
import scipy.io
import skimage.transform

from tensorflow.python.keras.applications.vgg16 import preprocess_input

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, counting_dataset, labels, ranking_dataset, batch_size=25, dim=(224,224), n_channels=3, shuffle=True, rank_images=5):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.counting_dataset = counting_dataset
        self.ranking_dataset = ranking_dataset
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.rank_images = rank_images
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.ranking_dataset) / self.rank_images)) # rank_images è il numero di immagini di ranking che prendo per ogni batch (da ognuna genero 5 sottopatch)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.rank_images:(index+1)*self.rank_images] # indici delle 5 immagini di ranking da cui creo il batch di 25 immagini
        
        # counting_indexes = random.sample(range(0, len(self.counting_dataset)), self.batch_size)
        counting_indexes = [random.randint(0, len(self.counting_dataset)-1) for _ in range(self.batch_size)]
          
        counting_dataset_temp = [self.counting_dataset[k] for k in counting_indexes]
        list_ranking_imgs_temp = [self.ranking_dataset[k] for k in indexes] 

        # Generate data
        X, y = self.__data_generation(counting_dataset_temp,list_ranking_imgs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.rank_images)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, counting_dataset_temp,list_ranking_imgs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X_counting = np.empty((self.batch_size, *self.dim, self.n_channels)) #counting batch
        X_ranking = np.empty((self.batch_size, *self.dim, self.n_channels)) #ranking batch
        y = np.empty((self.batch_size,14,14,1)) #counting batch target
        y_tmp = np.empty((self.batch_size,14,14)) #usato per salvare temporaneamente il resize del counting target
        
        #Generate counting batch
        for i, image in enumerate(counting_dataset_temp):
            img = cv2.imread(image)
            height, width = img.shape[:-1]            
            
            ####From PAPER: During training we sample one sub-image from each training image per epoch
            #From PAPER: To further improve the performance of our baseline network, we introduce multi-scale sampling from the available
            #labeled datasets. Instead of using the whole image as an input, we randomly sample square patches of varying size (from 56 to 448 pixels).
            random_size = random.randint(min(56,width,height), min(448,width,height))#select random size
            
            random_x1 = random.randint(0, width-random_size)#select random point (the left-top corner of the patch)
            random_y1 = random.randint(0, height-random_size)
            crop_img = img[random_y1:random_y1+random_size, random_x1:random_x1+random_size]            
            
            res = cv2.resize(crop_img,(224,224),interpolation = cv2.INTER_LINEAR)#INTER_LINEAR - a bilinear interpolation
            
            img_array = np.asarray(res)
            
            X_counting[i,] = img_array
            
            dmap = self.labels[image]
            crop_dmap = dmap[random_y1:random_y1+random_size, random_x1:random_x1+random_size]
            
            #resize to CNN output
            y_tmp[i] = skimage.transform.resize(crop_dmap, (14,14), anti_aliasing=True)
            y[i] = np.resize(y_tmp[i],(14,14,1))

        #### ALGORITHM TO GENERATE RANKED DATASETS ####
        k = 5 #number of patches
        s = 0.75 #scale factor
        r = 8  #anchor region            
        for i, image in enumerate(list_ranking_imgs_temp):            
            img = cv2.imread(image)
            height, width = img.shape[:-1]                    
            
            #select anchor region: crop a patch 1/r of original image centered in the same point and with same aspect ratio
            anchor_region_width = width * math.sqrt(1/r)
            anchor_region_height = height * math.sqrt(1/r)
            
            center_x = width/2
            center_y = height/2
            
            x1 = int(round(center_x - (anchor_region_width/2)))
            y1 = int(round(center_y - (anchor_region_height/2)))  
            x2 = int(round(center_x + (anchor_region_width/2)))
            y2 = int(round(center_y + (anchor_region_height/2)))
            
            #anchor region
            # crop_img = img[y1:y2, x1:x2]
                
            #STEP 1: choose an anchor point ramdomly from the anchor region
            anchor_point_x = random.uniform(x1, x2)
            anchor_point_y = random.uniform(y1, y2)
            
            #STEP 2: find the largest square patch centered at the anchor point and contained within the image boundaries
            anchor_point_offset_x = width - anchor_point_x
            anchor_point_offset_y = height - anchor_point_y
            
            coord_list = [anchor_point_x, anchor_point_y, anchor_point_offset_x, anchor_point_offset_y]
            patch1_half_width = min(coord_list)
                
            patch1_x1 = int(round(anchor_point_x - patch1_half_width))
            patch1_y1 = int(round(anchor_point_y - patch1_half_width))
            patch1_x2 = int(round(anchor_point_x + patch1_half_width))
            patch1_y2 = int(round(anchor_point_y + patch1_half_width))
            
            #first_patch
            crop_img1 = img[patch1_y1:patch1_y2, patch1_x1:patch1_x2]
            crop_list = list()
            crop_list.append(crop_img1)
            res0 = cv2.resize(crop_img1,(224,224),interpolation = cv2.INTER_LINEAR)
            res_array = np.asarray(res0)

            X_ranking[i*k,] = res_array
            
            #STEP 3: Crop k-1 additional square patches, reducing size iteratively by a scale factor s. Keep all patches centered at anchor point    
            for j in range(2, k+1):
                img = crop_list[j-2]
                h, w = img.shape[:-1]
                
                crop_width = w * math.sqrt(s) #same as height
                
                cen_x = w/2
                cen_y = h/2
                
                xy1 = int(round(cen_x - (crop_width/2)))
                xy2 = int(round(cen_x + (crop_width/2)))

                crop_img2 = img[xy1:xy2, xy1:xy2]
                crop_list.append(crop_img2)
                
                res_tmp = cv2.resize(crop_img2,(224,224),interpolation = cv2.INTER_LINEAR)
                img_array = np.asarray(res_tmp)

                X_ranking[(i*k)+j-1,] = img_array   
        y_ranking = np.zeros((self.batch_size,1,1,1)) # dummy ranking batch target
        X_counting = preprocess_input(X_counting)
        X_ranking = preprocess_input(X_ranking)
        return {'counting_input': X_counting, 'ranking_input': X_ranking}, {'conv2d_1': y, 'average_pooling2d_1': y_ranking}
