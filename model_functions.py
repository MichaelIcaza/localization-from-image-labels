"""
This file contains some model functions, such as visualizations and the data generator
"""
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt


#generator class for producing samples from the drive. 
class DataGenerator(Sequence):

    'Generates data for Keras'
    def __init__(self, image_dir, targets, shuffle, augment): #these files currently need to arrive pre-split for validaiton/folds
        #these are large images, batch size is fixed at size 1
        'Initialization'
        self.image_dir = image_dir
        self.targets = targets
        self.shuffle = shuffle
        self.augment = augment
        self.ix_seq = np.arange(np.shape(self.targets)[0])#will work differently with folds
        self.on_epoch_end()
        if self.shuffle==True:
            np.random.shuffle(self.ix_seq)
            
    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.targets)
    
    def __getitem__(self, index):
        'Generate one image of data'
        batch_ix = self.ix_seq[index]#exchange sequential call with shuffleable ix list
        file_loc = self.image_dir[batch_ix]#get the file-name parameters from csv file (pandas format)     
        x_img = np.zeros((1, 10, 800, 800, 4), dtype=np.uint16)
        y_tar = np.zeros((1, 1), dtype=np.bool)
        
        if self.augment:
            k = np.random.randint(0,3)#rotation multiple of 90deg
            t = np.random.randint(0,1)#transpose bool
            x_img = np.rot90(x_img, k=k, axes=(2, 3))
            if t:
                x_img = np.transpose(x_img, axes=(2,3))
                
        
        x_img[0] = self.__data_generation(file_loc)
        y_tar[0,0] = self.targets[batch_ix]

        return x_img, y_tar#none indexing adds the batch-size-dimension
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.ix_seq)#shuffle the order of lines

    def __data_generation(self, file_loc):
        'Generates one image slice'
        return np.load(file_loc)



class PlotLoss(Callback):
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
            fig, ax = plt.subplots(1,2,figsize=(5,2))
            ax[0].plot(self.loss, label='Loss')
            ax[0].plot(self.vloss, label='Val Loss')
            #ax[0].set_ylim(0,None)
            ax[0].legend()
            ax[1].plot(self.acc, label='Accuracy')
            ax[1].plot(self.vacc, label='Val Accuracy')
            #ax[1].set_ylim(0,1)
            ax[1].legend()
            plt.show()
