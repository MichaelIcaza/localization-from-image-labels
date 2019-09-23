"""
This file contains a single function for transforming the original .stk data into .npy samples, as well as a {file_name: class} dictionary

The rest of this project assumes the file structure is set up as shown below. 

The original data is structured as such:
/data
    /01
        [img1ch1].stk
        [img1ch2].stk
        ...
    /02
    ...

The function creates a new dictory as such
/numpy_data
    01-img1.npy
    01-img2.npy
    ...
    file_data.pkl {imagename:class} dict

The other files will assume the data is in the second format. 
    
"""


import os
import pickle as pkl
import tifffile as tiff #Imports "metamorph" .stk files

import numpy as np

def stk_to_npy(inp_dir, out_dir):
    #first, we try to create the output directory, if it already exists we don't do anything. 
    #This means if this process was aborted and restart, it may have some issues
    #although its more likely it will just overwite the old files
    try:
        os.mkdir(out_dir)
        print("numpy_data folder created")
    except:
        print("numpy_data folder already exists")

    
    #adjustable parameters if you're trying to canabalize this function
    img_slice_size = (10, 800, 800, 1)
    max_depth = img_slice_size[0]   
    
    #dictionary of patient ID (also, directory info) to Class ID (note, 1 is ChRCC diagnosed tissue samples)
    #normally this would be in a csv but I got it as text in an email so we're not going to talk about this. 
    file_data = {}#we'll collect name data for this as we open files
    patient_status = {'02':1,'04':1,'05':0,'06':0,'07':1,
                      '09':1,'11':1,'12':0,'13':0,'14':0,
                      '15':0,'16':1,'17':1,'18':1,'19':1,
                      '20':0,'21':1,'22':1,'23':0,'24':1,
                      '25':0,'26':0,'28':1,'29':0,'31':0}
    
    skip_terms = ["Color", "region"]#if these terms are in the file name, do not process

    for patient_id in patient_status.keys():#we visit the patient id folders (this skips over some patients in the original dataset)
        patient_dir = inp_dir+"/"+patient_id+"/"#we append patient ID, this is how the original file structure was set up
        for filename in sorted(os.listdir(patient_dir)): #This assumes alphabetical sorting is properly separting the different images and sorting the channels. Sequential or date-time info in image names can mess with this assumption
            if any(term in filename for term in skip_terms):#checks if any skipterms are in the file name
                continue#We aren't using the "Color Combine" images
            if "zoom2_kalman3" in filename:#As mentioned before, we take the higher resolution lower depth of field images, ignoring the other images 
                channel = int(str(filename)[-5]) #neg 5 is an offset to grab the interger before ".tiff", the channel number 1..4
                channel -=1 #transform from the 1..4 ch # to the 0..3 index value
                with tiff.TiffFile(patient_dir+filename) as tif:
                    img = tif.asarray().astype(np.uint16)
                    
                    #Crops/pads layers to size 10
                    if np.shape(img)[0] > max_depth:
                        img = img[:max_depth,:,:]#crops if too many layers
                        print('oversized image detected: ', filename)#the process will continue, but only the first 10 stacks (depth layers) will be processed
                    if np.shape(img)[0] < max_depth:
                        a = max_depth-np.shape(img)[0]#depth padding amount 
                        img = np.pad(img, ((0,a),(0,0),(0,0)), mode='constant', constant_values=(0.,0.))#post-padding
                    
                    #this is where we assume we're getting sequential channels of the same image
                    img = np.reshape(img, img_slice_size) #an error here indicates the image is probably of the wrong size
                    if channel ==0:image=img #When channel is 0, create a new "image" array
                    if channel > 0:image=np.concatenate((image, img), axis=3)#Last axis is for channels
                    if channel ==3:#if channel is 3, we assume the image is completed (after the previous step is done)
                        img_name = patient_id+"_"+filename[:-8]#-8 offsets to remove CH#.STK from file name
                        np.save(out_dir+"/"+img_name+".npy", image)#saves the image as npy
                        file_data[img_name] = patient_status[patient_id] #retrieve the binary class, put it in the dictionary
    #numpy doesn't like saving dictionaries, so we pickle it
    with open(out_dir+'/image_classes.pkl', 'wb') as cucumber:
        pkl.dump(file_data, cucumber)
