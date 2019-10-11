"""
This file contains some model functions, such as visualizations and the data generator
"""
from keras.utils import Sequence
from keras.callbacks import Callback
from keras.optimizers import Optimizer
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import matplotlib.patches as patches
from model import build_model


class Plot_Predictions(Callback):
    def __init__(self, x_val, y_val, ix):
        self.x = x_val
        self.y = y_val
        self.ix = ix
        self.map2xy
        self.plot_max
        
    def on_epoch_end(self, epoch, logs={}):
        reg_model = build_model("heat_map", [16, 32, 32, 64, 16], droprate=0.25, optimizer="Nadam", lr=1e-5, grad_accum=1)
        reg_model.set_weights(self.model.get_weights())
        heat_maps = reg_model.predict(self.x[self.ix])
        self.plot_max(heat_maps, ch=3, n=5)

    
    def map2xy(self, map_i, n):#inputs one map prediction (17,17,1)
        xy = []
        probs = []
        field = np.copy(map_i[:,:,0])
        for patch in range(n):
            prob = np.max(field).round(3)
            probs.append(prob)
            y, x = np.unravel_index(field.argmax(), field.shape)
            xy.append((20*x,20*y)) #for upper-left xy for Rectangle()
            field[y,x]=0#set the value to -1 so we can find the next max
        del field
        return xy, probs
    
    def plot_max(self,maps,ch, n):
        plt.gray()
        colors = ['r', 'm', 'y', 'g', 'b', 'c']
        fig,ax = plt.subplots(2,3)
        for i in range(len(maps)):
            a=i//3
            b=i%3
            arr = np.max(self.x[i,:,:,:,ch], axis=0)
            ax[a,b].imshow(arr)
            xy, probs = self.map2xy(maps[i,:,:,:],n)
            for j in range(len(xy)):
                if probs[j]>=0.5:
                    ax[a,b].add_patch(patches.Rectangle(xy[j],80,80,edgecolor=colors[j], fill=False))
                    font = {'color':colors[j], 'size':12,}
                    ax[a,b].text(xy[j][0],xy[j][1], str(round(probs[j],3)), fontdict=font)
                    ax[a,b].get_xaxis().set_visible(False)
                    ax[a,b].get_yaxis().set_visible(False)
        plt.show()
            #for i in range(len(probs)):print(colors[i]+':'+str(probs[i]))







class roc_callback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x, batch_size=1)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val, batch_size=1)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return
    
    
#generator class for producing samples from system RAM 
#This data generator requires around 8-10GB of system memory to run on my system.
class DataGenerator(Sequence):
    'Feed me Numpy Arrays'
    def __init__(self, x_data, targets, batch_size, shuffle, augment, weighted): #these files currently need to arrive pre-split for validaiton/folds
        #these are large images, batch size is fixed at size 1
        'Initialization'
        self.x_data = x_data
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.weighted = weighted
        
        self.ix_seq = np.arange(np.shape(self.targets)[0])#for shuffling, this can be shuffled with out messing with sampling
        self.on_epoch_end()
        #shuffle before training (and after each epoch)
        if self.shuffle:    
            np.random.shuffle(self.ix_seq)    
        #class weight
        if self.weighted:#the weights are determined by the entire train set, or train fold, not the output batch (which is probably 1, will cause div-0 issues)
            pos = np.sum(self.targets)
            neg = np.sum(1-self.targets)
            self.class_weights = {0:pos/neg, 1:neg/pos}
            print(self.class_weights)
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.targets)//self.batch_size
    
    def __getitem__(self, index):
        'Generate one image of data'
        ixs = self.ix_seq[index*self.batch_size:(index+1)*self.batch_size]#exchange sequential call with shuffleable ix list
        x_batch = np.zeros((len(ixs),10,400,400,4), dtype=np.uint16)#if the remaining samples is less than self.batch_size, len(ixs) will be the shorter correct value, but using self.batch_size here would cause "sample" padding, sending zero-ed images into the network
        y_batch = np.zeros((len(ixs),1), dtype=np.bool)

        for i, ix in enumerate(ixs):
            x_batch[i], y_batch[i] = self.__data_generation(ix)
        sample = (x_batch, y_batch)
        
        if self.weighted:#if weighted, append the sample_weight to the sample tuple
            sample_weight = np.asarray([self.class_weights[target] for target in y_batch[:,0]])#we need to reduce the dim of y batch so it returns 0 or 1, not [0], or [1]
            sample += (sample_weight,)
            
        return sample
    
    def __data_generation(self, ix):
        x_sample = self.x_data[ix:ix+1]#np.zeros((1, 10, 800, 800, 4), dtype=np.uint16)
        y_sample = self.targets[ix:ix+1,None]#none in the index adds a dimension size 1
        
        if self.augment:
            #these augments only do flip and rotate transforms
            k = np.random.randint(0,3)#rotation multiple of 90deg
            t = np.random.randint(0,1)#transpose bool
            x_sample = np.rot90(x_sample, k=k, axes=(2,3))
            if t:x_sample = np.transpose(x_sample, axes=(2,3))
            #note: there are additional preprocessing methods we can use, but the extrapolation is tricky, by keeping
            #the data in uint format, we keep full precision and can send data to the GPU faster. These transforms can be thought of as index-transforms, no pixel interpolation. 
        return x_sample, y_sample
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.ix_seq)#shuffle the order of samples via the index lookup rather than literal shuffling
            

        
#generator class for producing samples from the drive, basically the same as above
#if you have ~8GB of memory, the normal generator may not work so here is a generator that streams from drive storage. 
#rather than feeding this generator image data, feed it the file locations and it will open them (assumped .npy format)
#this method is about 15% slower. 
class StreamDataGenerator(Sequence):    
    'Feed me File Locations'
    def __init__(self, image_dir, targets, shuffle, augment, weighted): #these files currently need to arrive pre-split for validaiton/folds
        #these are large images, batch size is fixed at size 1
        'Initialization'
        self.image_dir = image_dir
        self.targets = targets
        self.shuffle = shuffle
        self.augment = augment
        self.weighted = weighted
        
        self.ix_seq = np.arange(np.shape(self.targets)[0])#for shuffling, this can be shuffled with out messing with sampling
        self.on_epoch_end()
        #shuffle before training (and after each epoch)
        if self.shuffle==True:    
            np.random.shuffle(self.ix_seq)    
        #class weight
        if self.weighted:
            pos = np.sum(self.targets)
            neg = np.sum(1-self.targets)
            self.class_weights = {0:pos/neg, 1:neg/pos}
            print(self.class_weights)
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.targets)
    
    def __getitem__(self, index):
        'Generate one image of data'
        batch_ix = self.ix_seq[index]#exchange sequential call with shuffleable ix list
        file_loc = self.image_dir[batch_ix]#get the file-name parameters from csv file (pandas format)     
        x_img = np.zeros((1, 10, 400, 400, 3), dtype=np.uint16)
        y_tar = np.zeros((1, 1), dtype=np.bool)
        sample_weight = np.zeros((1,), dtype=np.float32)
        
        if self.augment:
            k = np.random.randint(0,3)#rotation multiple of 90deg
            t = np.random.randint(0,1)#transpose bool
            x_img = np.rot90(x_img, k=k, axes=(2, 3))
            if t:x_img = np.transpose(x_img, axes=(2,3))
                
        
        x_img[0] = self.__data_generation(file_loc)
        y_tar[0,0] = self.targets[batch_ix]    
        sample = (x_img, y_tar)
        
        if self.weighted:#if weighted, append the sample_weight to the sample tuple
            sample_weight = np.asarray([self.class_weights[self.targets[batch_ix]]])
            sample += (sample_weight,)
        return sample
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.ix_seq)#shuffle the order of samples via the index lookup rather than literal shuffling

    def __data_generation(self, file_loc):
        'Generates one image slice'
        return np.load(file_loc)

class PlotLoss(Callback):
    #this plots the loss every "freq" epochs, assumes "accuracy" metric
    def __init__(self, loss=[], vloss=[], acc=[], vacc=[], freq=1):
        self.loss = []
        self.vloss = []
        self.acc = []
        self.vacc = []
        self.freq = freq
    def on_epoch_end(self, epoch, logs={}):
        self.loss.append(logs.get('loss'))
        self.vloss.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.vacc.append(logs.get('val_acc'))
        if (epoch+1)%self.freq==0:
            fig, ax = plt.subplots(1,2,figsize=(8,2))
            ax[0].plot(self.loss, label='Loss')
            ax[0].plot(self.vloss, label='Val Loss')
            ax[0].legend()
            ax[1].plot(self.acc, label='Accuracy')
            ax[1].plot(self.vacc, label='Val Accuracy')
            ax[1].legend()
            plt.show()
            
class PlotHist(Callback):
    #this plots the loss every "freq" epochs, assumes "accuracy" metric
    def __init__(self, map_model, val_gen, val_len, val_tar, freq):
        self.map_model = map_model
        self.val_gen = val_gen
        self.val_len = val_len
        self.val_tar = val_tar
        self.freq = freq
    def on_epoch_end(self, epoch, logs={}):
        if (epoch+1)%self.freq==0:
    
            temp_weights = self.model.get_weights()
            #get prediction image stats        
            self.map_model.set_weights(temp_weights)
            
            val_map = self.map_model.predict_generator(self.val_gen, self.val_len)
            fig, ax = plt.subplots(1,2, figsize=(8,2))
            ax[0].hist(val_map[self.val_tar==0].flatten(),bins=20, histtype='step', label="negative")
            ax[0].hist(val_map[self.val_tar==1].flatten(),bins=20, histtype='step', label="positive")
            ax[0].legend()
            
            ax[1].hist(np.max(val_map[self.val_tar==0], axis=(1,2,3)).flatten(), bins=5, histtype='step', label="negative")
            ax[1].hist(np.max(val_map[self.val_tar==1], axis=(1,2,3)).flatten(), bins=5, histtype='step', label="positive")
            ax[1].legend()
            
            plt.show()
            
     
