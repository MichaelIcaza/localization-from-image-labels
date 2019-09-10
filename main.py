"""
This file runs the main experiment. 
Note, this procedure returns a tensorflow warning about large droprate. This is not an issue, we intend to use an abnormally large dropout rate


"""
data_dir = "E:/Wu/numpy_data"


import pickle as pkl
import numpy as np

from model_functions import DataGenerator, PlotLoss
from model import build_model
from tensorflow.keras.callbacks import LearningRateScheduler


from sklearn.model_selection  import StratifiedKFold

#======================== Load Data Files
with open(data_dir+"/image_classes.pkl", 'rb') as cucumber:
    img_dict = pkl.load(cucumber)



#======================== Begin Stratified K-Fold
#prepare for splitting
img_names = np.asarray(list(img_dict.keys()))#get the names
img_tar = np.asarray([img_dict[img] for img in img_names])#get the boolean class
img_loc = np.asarray([data_dir+'/'+name+'.npy' for name in img_names])#turn names into file_locations by adding the file_path as suffix

#split the data into folds, by patient
patient_classes = {name[:2]:img_dict[name] for name in img_names}
#create two lists, [patient ID], and [target_class] to split on
patient_list, target_list = [],[]
for k, v in patient_classes.items():
    patient_list.append(k)
    target_list.append(v)
patient_list = np.asarray(patient_list)
target_list = np.asarray(target_list)

#sk-learn's staratified K-fold splits while keeping fold balanced about the target
skfold = StratifiedKFold(n_splits=5, random_state=1)
folds = list(skfold.split(patient_list, target_list))

#we need to translate the patient-level folds to image-location folds
train_id_ix = list(folds[0][0])
train_ids = [patient_list[ix] for ix in train_id_ix]
train_ix = [name[:2] in train_ids for name in img_names]
val_id_ix = list(folds[0][1])
val_ids = [patient_list[ix] for ix in val_id_ix]
val_ix = [name[:2] in val_ids for name in img_names]

#we now have the folds at the image-level
train_loc = img_loc[train_ix]
train_tar = img_tar[train_ix]
train_len = len(train_loc)
val_loc = img_loc[val_ix]
val_tar = img_tar[val_ix]
val_len = len(val_loc)


#=================Begin Training

train_generator = DataGenerator(train_loc, train_tar, shuffle=True, augment=True)
val_generator = DataGenerator(val_loc, val_tar, shuffle=False, augment=False)
droprate=0.8
lr = 1e-5
model = build_model(out_type="class", droprate=droprate, optimizer="Nadam", lr=lr)#the model trains in "class mode", and can output a "heat_map" in later analysis

def schedule(epoch):
    lr_list = np.logspace(-6, -4, 5)
    ix = epoch//5#this value determines how many epochs before adjusting lr
    if ix>=len(lr_list):
        return lr_list[-1]    
    else:
        return lr_list[ix]

lr_sched = LearningRateScheduler(schedule, verbose=1)
plot_loss = PlotLoss(freq=5)

model.fit_generator(train_generator, steps_per_epoch=train_len, epochs=1000,
                    validation_data=val_generator, validation_steps=val_len,
                    callbacks=[plot_loss], 
                    max_queue_size=128, workers=16)
