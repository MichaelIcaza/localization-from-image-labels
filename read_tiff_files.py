#See line 38-ish to adjust the directory where this file is located (variable: "here")
import os
import math as m
import tifffile as tiff #Imports "metamorph" .stk files
from skimage.measure import block_reduce

import numpy as np
np.random.seed = 123

#This function gets opens the pre-determined directory 
def load_stk(file_dir, depth_padding=10): #Maximum depth appears to be 10
    stack = []
    names = []
    for filename in os.listdir(file_dir): #Dependent on the only files in folders being image data files
        if "zoom2" in filename:
            channel = int(str(filename)[-5])-1 #Grabs channel number, from 1-4 to 0-3
            with tiff.TiffFile(file_dir+filename) as tif:
                img = tif.asarray()
                if np.shape(img)[0] > 10:
                    img = img[:10,:,:]
                if np.shape(img)[0] < 10:#Checks layer num in order to pad to 10
                    a = depth_padding-np.shape(img)[0] 
                    img = np.pad(img, ((0,a),(0,0),(0,0)), mode='constant', constant_values=(0.,0.))
                img = np.reshape(img, (10, 800, 800, 1)) #an error here indicates the image is probably of the wrong size, depth >10 or not 800x800
                img_smol = block_reduce(img, (1,2,2,1), np.average)
                #Now to combine images into channels
                if channel ==0:image=img_smol #When channel is 0, create a new "image" array
                if channel > 0:image=np.concatenate((image, img_smol), axis=3)#Last axis is for channels
                if channel ==3:
                    stack.append(image) #On finding the last channel, append to list
                    names.append(filename[:-8])
    return stack, names

    
#The below assumes this file is in a "model" folder, parallel to "merged" folder
def load_data():
    here = os.path.dirname(__file__)

#    try:
#        here = os.path.dirname(__file__)
#    except:
#        here = r"C:\Users\Michael\Desktop\Wu\_model"
        
    ch_files = os.path.join(here, ".././merged/ch/")
    on_files = os.path.join(here, ".././merged/on/")
    
    ch, ch_names = load_stk(ch_files)
    on, on_names = load_stk(on_files)
    
    on= np.asarray(on)
    ch = np.asarray(ch)
    
    ch_y = np.ones(len(ch), dtype=int)
    on_y = np.zeros(len(on), dtype=int)
    

    xdata = np.vstack((on,ch)) #merge x and y into vars
    ydata = np.hstack((on_y,ch_y))
    xnames = np.asarray(on_names+ch_names)
    return xdata, ydata, xnames