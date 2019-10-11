"""
This file generates sample images, no CV
"""
import os
import time
import pickle as pkl
import numpy as np
import pandas as pd
from model import build_model
from model_functions import DataGenerator, PlotLoss, PlotHist, roc_callback, Plot_Predictions
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection  import StratifiedKFold

data_dir = "../NumpyData_1x"
trial_dir = time.strftime("%Y-%m-%d_%H-%M/")
os.mkdir(trial_dir)#create folder for weights, labeled by date-time

#Outline:
#   open meta data files, names targets and locations that we use to split data and import the samples correctly
#   Then, we split the data using SKLearn's stratified kfold 
#   then we run the main experiment, run various dropout rates at differnt folds
#   we write files over every trial (unique dropout rate on 5 folds)

#======================== Load Meta Data Files
with open(data_dir+"/image_classes.pkl", 'rb') as cucumber:
    img_dict = pkl.load(cucumber)

#adjust the files into proper format for the generator, we want arrays
img_names = np.asarray(list(img_dict.keys()))#get the names
img_tar = np.asarray([img_dict[img] for img in img_names])#get the boolean target class
img_loc = np.asarray([data_dir+'/'+name+'.npy' for name in img_names])#turn names into file_locations by adding the file_path as suffix

#split the data into folds, by patient
patient_class_dict = {name[:2]:img_dict[name] for name in img_names}#used to lookup the image's target value
patient_list, target_list = [],[]#create two lists, [patient ID], and [target_class] to split on
for k, v in patient_class_dict.items():#go over everything, order doesn't matter
    patient_list.append(k)
    target_list.append(v)
patient_list = np.asarray(patient_list)
target_list = np.asarray(target_list)

#======================== Split patients into folds
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=128)#sk-learn's staratified K-fold splits while keeping fold balanced about the target
folds = list(skfold.split(patient_list, target_list))

#save an early copy of the fold data, for analysis mostly
fold_arr = np.zeros((len(patient_list),2))
for i, patient in enumerate(patient_list):
    for f, fold in enumerate(folds):
        if i in fold[1]:
            fold_arr[i,0] = patient
            fold_arr[i,1] = f
fold_arr = np.asarray(fold_arr)
patient_fold_dict = {int(row[0]):int(row[1]) for row in fold_arr}
fold_df = pd.DataFrame({"Patient_ID":fold_arr[:,0], "Fold":fold_arr[:,1]})
fold_df.to_csv(trial_dir+"fold_data.csv", index=False)#if index is true, a 0th column is added that just has row number
        

#=================================Begin the experiment

train_id_ix = list(folds[1][0])
train_ids = [patient_list[ix] for ix in train_id_ix]
train_ix = [name[:2] in train_ids for name in img_names]
train_loc = img_loc[train_ix]#list of image names in train set
train_tar = img_tar[train_ix]
train_len = len(train_loc)


val_id_ix = list(folds[1][1])
val_ids = [patient_list[ix] for ix in val_id_ix]
val_ix = [name[:2] in val_ids for name in img_names]
val_loc = img_loc[val_ix]#list of image names in validation set
val_tar = img_tar[val_ix]
val_len = len(val_loc)

#open all of the train-fold images
x_train = np.asarray([np.load(file) for file in train_loc])
x_val = np.asarray([np.load(file) for file in val_loc])
    
train_generator = DataGenerator(x_train, train_tar, 1, shuffle=True, augment=True, weighted=False)
val_generator = DataGenerator(x_val, val_tar, 1, shuffle=False, augment=False, weighted=False)

#fixed parameters
sizes = [16, 32, 32, 64, 16]
lr = 1e-5
optimizer = "Nadam"
droprate=0.25

weight_name = trial_dir+str(int(droprate*100))+"_"+str(i+1)+"fold_weights.hdf5"

model = build_model(out_type="class", sizes=sizes, droprate=droprate, optimizer=optimizer, lr=lr, grad_accum=1)#the model trains in "class mode", and can output a "heat_map" in later analysis
map_model = build_model(out_type="heat_map", sizes=sizes, droprate=droprate, optimizer=optimizer, lr=lr, grad_accum=1)

#model.summary()#this will print out model details including layer sizes and parameter count ~100K

#load the same initial random weights for each trial

if os.path.exists(trial_dir+"initial_weights.hdf5"):#if the initial weights exist, load. otherwise, save
    model.load_weights(trial_dir+"initial_weights.hdf5")
else:
    model.save_weights(trial_dir+"initial_weights.hdf5")

plot_loss = PlotLoss(freq=10)#plots training metrics
plot_hist = PlotHist(map_model, val_generator, val_len, val_tar,10)#plots histogram of activations
plot_preds=Plot_Predictions(x_val, val_tar, [5,15,23,2,9,18])#plot predictions with bounding boxes
aucroc_callback = roc_callback((x_train, train_tar),(x_val, val_tar))#print out AUCROC metric
earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=25, verbose=0)#note: normally validation loss would be selected, we have noisy validation loss so we select this method

checkpoint = ModelCheckpoint(weight_name, monitor='loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)#we do not check the validation loss, its technically our "test set", its invisible during training        
hist = model.fit_generator(train_generator, steps_per_epoch=train_len, epochs=300,
                    validation_data=val_generator, validation_steps=val_len,
                    callbacks=[plot_loss, checkpoint, plot_hist, aucroc_callback, earlystop, plot_preds],
                    max_queue_size=128, workers=16)
    
   
model.load_weights(weight_name)#reload the best weights, according to monitored training loss
