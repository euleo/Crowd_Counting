import numpy as np
import tensorflow as tf
import cv2
import random
import math
import scipy.io
import PIL
import matplotlib.pyplot as plt
import os
from tensorflow.python.keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.preprocessing import image

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
        return int(np.floor(len(self.ranking_dataset) / self.rank_images)) # rank_images is the number of ranking images to take to create a ranking batch generating 5 sub-patches from each one

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.rank_images:(index+1)*self.rank_images] # indexes of the ranking images from which to generate the ranking batch
        
        # counting_indexes = random.sample(range(0, len(self.counting_dataset)), self.batch_size)
        counting_indexes = [random.randint(0, len(self.counting_dataset)-1) for _ in range(self.batch_size)] # indexes of the counting images from which to generate the counting batch
          
        counting_dataset_temp = [self.counting_dataset[k] for k in counting_indexes]
        list_ranking_imgs_temp = [self.ranking_dataset[k] for k in indexes] 

        # Generate data
        X, y = self.__data_generation(counting_dataset_temp,list_ranking_imgs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.ranking_dataset)) #inizializza sel.indexes
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, counting_dataset_temp,list_ranking_imgs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X_counting = np.empty((self.batch_size, *self.dim, self.n_channels)) #counting batch
        X_ranking = np.empty((self.batch_size, *self.dim, self.n_channels)) #ranking batch
        y_counting = np.empty((self.batch_size,14,14,1)) #counting batch target
        y_tmp = np.empty((self.batch_size,14,14)) # to temporarily save the resized counting target
        
        #Generate counting batch
        for i, image_path in enumerate(counting_dataset_temp):
            counting_img = image.load_img(image_path)            
            width, height = counting_img.size              
            
            ####From PAPER: During training we sample one sub-image from each training image per epoch
            #From PAPER: To further improve the performance of our baseline network, we introduce multi-scale sampling from the available
            #labeled datasets. Instead of using the whole image as an input, we randomly sample square patches of varying size (from 56 to 448 pixels).
            random_size = random.randint(min(56,width,height), min(448,width,height))#select random size
            
            random_x1 = random.randint(0, width-random_size)#select random point (the left-top corner of the patch)
            random_y1 = random.randint(0, height-random_size)
            
            crop_img = counting_img.crop((random_x1, random_y1, random_x1+random_size, random_y1+random_size)) 
            crop_resized_img = crop_img.resize((224,224),PIL.Image.BILINEAR)            
            crop_resized_array_img = image.img_to_array(crop_resized_img)            
            crop_resized_array_preproc_img = preprocess_input(crop_resized_array_img)            
            X_counting[i,] = crop_resized_array_preproc_img
            
            dmap = self.labels[image_path]            
            crop_dmap = dmap[random_y1:random_y1+random_size, random_x1:random_x1+random_size]            
            #resize to CNN output

            # I need this on my machine to not use deprecated imresize. Set ADB=False if you ger an error.
            ADB = True
            if ADB:
                import skimage.transform
                y_tmp[i] = skimage.transform.resize(crop_dmap, (14,14), anti_aliasing=True)
                y_counting[i] = np.resize(y_tmp[i],(14,14,1))
            else:
                y_tmp[i] = scipy.misc.imresize(crop_dmap, (14,14), interp='bilinear')           
                y_counting[i] = np.resize(y_tmp[i],(14,14,1))

            #### uncomment to save images as png to check if they are right 
            # newpath = 'Newfolder\\'+image_path
            # if not os.path.exists(newpath):
                # os.makedirs(newpath)            
            # count_img_name = image_path.replace('counting_data_UCF\\','')
            # count_img_name = count_img_name.replace('.jpg','')
            # plt.imsave(newpath+'\\'+count_img_name+'(1)original_img.png', counting_img)
            # plt.imsave(newpath+'\\'+count_img_name+'(2)crop_img.png', crop_img)   
            # plt.imsave(newpath+'\\'+count_img_name+'(3)crop_resized_img.png', crop_resized_img) 
            # plt.imsave(newpath+'\\'+count_img_name+'(4)crop_resized_array_img.png', crop_resized_array_img, format="png")
            # plt.imsave(newpath+'\\'+count_img_name+'(5)crop_resized_array_preproc_img.png', crop_resized_array_preproc_img, format="png")
            # plt.imsave(newpath+'\\'+count_img_name+'(6)dmap.png', dmap, cmap='jet')             
            # plt.imsave(newpath+'\\'+count_img_name+'(7)crop_dmap.png', crop_dmap, cmap='jet') 
            # plt.imsave(newpath+'\\'+count_img_name+'(8)y_tmp.png', y_tmp[i], cmap='jet')  
            # plt.imsave(newpath+'\\'+count_img_name+'(9)y_counting.png', y_counting[i][:,:,0], cmap='jet')    

            #### uncomment to save arrays in a text file to check if they are right            
            # np.savetxt(newpath+'\\'+count_img_name+'(4)crop_resized_array_img0.txt',crop_resized_array_img[:,:,0])
            # np.savetxt(newpath+'\\'+count_img_name+'(5)crop_resized_array_preproc_img0.txt',crop_resized_array_preproc_img[:,:,0])      
            # np.savetxt(newpath+'\\'+count_img_name+'(4)crop_resized_array_img1.txt',crop_resized_array_img[:,:,1])
            # np.savetxt(newpath+'\\'+count_img_name+'(5)crop_resized_array_preproc_img1.txt',crop_resized_array_preproc_img[:,:,1])
            # np.savetxt(newpath+'\\'+count_img_name+'(4)crop_resized_array_img2.txt',crop_resized_array_img[:,:,2])
            # np.savetxt(newpath+'\\'+count_img_name+'(5)crop_resized_array_preproc_img2.txt',crop_resized_array_preproc_img[:,:,2])            
            # np.savetxt(newpath+'\\'+count_img_name+'(6)dmap.txt',dmap)
            # np.savetxt(newpath+'\\'+count_img_name+'(7)crop_dmap.txt',crop_dmap)
            # np.savetxt(newpath+'\\'+count_img_name+'(8)y_tmp.txt',y_tmp[i])
            # np.savetxt(newpath+'\\'+count_img_name+'(9)y_counting.txt',y_counting[i][:,:,0])
           
        #### ALGORITHM TO GENERATE RANKED DATASETS ####
        k = 5 #number of patches
        s = 0.75 #scale factor
        r = 8  #anchor region            
        for i, image_path in enumerate(list_ranking_imgs_temp):            
            img = image.load_img(image_path)
            width, height = img.size            
            
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
            crop_patch = img.crop((patch1_x1, patch1_y1, patch1_x2, patch1_y2)) 
            crop_resized_patch = crop_patch.resize((224,224),PIL.Image.BILINEAR)
            crop_resized_array_patch = image.img_to_array(crop_resized_patch)            
            crop_resized_array_preproc_patch = preprocess_input(crop_resized_array_patch)     
            X_ranking[i*k,] = crop_resized_array_preproc_patch
            
            patches_list = list()
            patches_list.append(crop_patch)

            #### uncomment to save images as png to check if they are right
            # newpath = 'Newfolder\\'+image_path
            # if not os.path.exists(newpath):
                # os.makedirs(newpath)
            # rank_img_name = image_path.replace('ranking_data\\','')
            # rank_img_name = rank_img_name.replace('.jpg','')            
            # plt.imsave(newpath+'\\'+rank_img_name+'(1)original_img.png', img)                       
            # plt.imsave(newpath+'\\'+rank_img_name+'(2)crop_resized_patch.png', crop_resized_patch)
            # plt.imsave(newpath+'\\'+rank_img_name+'(3)crop_resized_array_patch.png', crop_resized_array_patch, format="png")
            # plt.imsave(newpath+'\\'+rank_img_name+'(4)crop_resized_array_preproc_patch-1.png', crop_resized_array_preproc_patch, format="png")            
            
            #STEP 3: Crop k-1 additional square patches, reducing size iteratively by a scale factor s. Keep all patches centered at anchor point    
            for j in range(2, k+1):
                patch = patches_list[j-2]
                w, h = patch.size
                
                crop_width = w * math.sqrt(s) #same as height
                
                cen_x = w/2
                cen_y = h/2
                
                xy1 = int(round(cen_x - (crop_width/2)))
                xy2 = int(round(cen_x + (crop_width/2)))

                crop_subpatch = patch.crop((xy1, xy1, xy2, xy2)) 
                patches_list.append(crop_subpatch)
                
                crop_resized_subpatch = crop_subpatch.resize((224,224),PIL.Image.BILINEAR)                
                crop_resized_array_subpatch = image.img_to_array(crop_resized_subpatch)               
                crop_resized_array_preproc_subpatch = preprocess_input(crop_resized_array_subpatch)
                X_ranking[(i*k)+j-1,] = crop_resized_array_preproc_subpatch

                #### uncomment to save images as png to check if they are right
                # plt.imsave(newpath+'\\'+rank_img_name+'(5)crop_resized_array_subpatch-'+str(j)+'.png', crop_resized_array_subpatch, format="png")
                # plt.imsave(newpath+'\\'+rank_img_name+'(6)crop_resized_array_preproc_subpatch-'+str(j)+'.png', crop_resized_array_preproc_subpatch, format="png")
        y_ranking = np.zeros((self.batch_size,1)) # dummy ranking batch target
        X_counting = preprocess_input(X_counting)
        X_ranking = preprocess_input(X_ranking)
        return {'counting_input': X_counting, 'ranking_input': X_ranking}, {'counting_output': y_counting, 'ranking_output': y_ranking}

