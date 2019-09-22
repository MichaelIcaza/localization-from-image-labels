"""
This file runs the main experiment. 
Note, this procedure returns a tensorflow warning about large droprate. This is not an issue, we intend to use an abnormally large dropout rate


"""
import os
import time
import pickle as pkl
import numpy as np
import pandas as pd
from model_functions import DataGenerator, PlotLoss
from model import build_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection  import StratifiedKFold

data_dir = "E:/Wu/numpy_data"
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
skfold = StratifiedKFold(n_splits=5, random_state=1)#sk-learn's staratified K-fold splits while keeping fold balanced about the target
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
for droprate in [0.25, 0.85, 0, 0.5, 0.75, 0.9]:#itterate over the drop rates
    experiment_log=[]#this holds a list of the other logs, collected for every different drop rate
    trial_preds = {}#this holds img_name:prediction, added to over each fold
    for i in range(5):#itterate over the folds
        hist_log = []#training log from keras
        pred_log = []#predictions at end of training
        sample_log = []#names of the samples ([train],[val]) format

        #we have the patient-level folds, we get the iamge-level samples
        train_id_ix = list(folds[i][0])
        train_ids = [patient_list[ix] for ix in train_id_ix]
        train_ix = [name[:2] in train_ids for name in img_names]
        val_id_ix = list(folds[i][1])
        val_ids = [patient_list[ix] for ix in val_id_ix]
        val_ix = [name[:2] in val_ids for name in img_names]
        
        #split the train/val data by image name
        train_loc = img_loc[train_ix]#list of image names in train set
        train_tar = img_tar[train_ix]
        train_len = len(train_loc)
        val_loc = img_loc[val_ix]#list of image names in validation set
        val_tar = img_tar[val_ix]
        val_len = len(val_loc)
        
        #open all of the train-fold images
        x_train = np.asarray([np.load(file) for file in train_loc])
        x_val = np.asarray([np.load(file) for file in val_loc])
        
        train_generator = DataGenerator(x_train, train_tar, 1, shuffle=True, augment=True, weighted=True)
        val_generator = DataGenerator(x_val, val_tar, 1, shuffle=False, augment=False, weighted=False)
        
        #this is to account for the batch size
        
        sizes = [16, 32, 32, 64, 16]
        lr = 1e-4
        optimizer = "Adam"
    
        model = build_model(out_type="class", sizes=sizes, droprate=droprate, optimizer=optimizer, lr=lr, grad_accum=1)#the model trains in "class mode", and can output a "heat_map" in later analysis
        weight_name = trial_dir+str(int(droprate*100))+"_"+str(i+1)+"fold_weights.hdf5"
        #model.summary()#this will print out model details including layer sizes and parameter count ~100K
        plot_loss = PlotLoss(freq=10)
        #earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=0)#note: normally validation loss would be selected, we have noisy validation loss so we select this method
        checkpoint = ModelCheckpoint(weight_name, monitor='loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)#we do not check the validation loss, its technically our "test set", its invisible during training        
        hist = model.fit_generator(train_generator, steps_per_epoch=train_len, epochs=50,
                            validation_data=val_generator, validation_steps=val_len,
                            callbacks=[plot_loss, checkpoint],
                            max_queue_size=128, workers=16)
        model.load_weights(weight_name)#reload the best weights, according to monitored training loss
        preds = model.predict_generator(val_generator, val_len)
        
        #append to the fold-log (each of these should hold 5 items)
        sample_log.append((train_loc, val_loc))
        hist_log.append(hist.history)
        pred_log.append(preds)
        
        for n, pred in enumerate(preds):#for each pred, get the loc, extract the name, create dict name>pred
            img_name = val_loc[n].split('/')[-1][:-4]
            trial_preds[img_name] = pred[0]#we do this over every fold and eventually get a (validation/test) prediction for every image, the [0] makes it return a float rather than a [0.2413] number
            
        del model#to reset weights
    experiment_log.append([droprate, sample_log, hist_log, pred_log])#append to a list, including the drop rate. 
    #a complete 5-fold validation has been done
    #save as a pickle for later analysis, for every trial
    str_drop=str(int(droprate*100))
    with open(trial_dir+str_drop+'.pkl', 'wb') as f:
        pkl.dump(experiment_log, f)
    
    #also write a csv for ongoing-analysis while experiment is running
    #columns: File name, patient_id, Fold, Target, Pred
    id_data = [name[:2] for name in img_names]
    fold_data = [patient_fold_dict[int(id_)] for id_ in id_data]
    pred_data = [trial_preds[name] for name in img_names]
    trial_df = pd.DataFrame({"File_Name":img_names, "Patient_ID":id_data, "Fold":fold_data, "Target":img_tar, "Predicted":pred_data})
    trial_df.to_csv(trial_dir+str_drop+"_predictions.csv", index=False)
    
    
            


