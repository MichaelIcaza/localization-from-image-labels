"""The Experiment file holds functions for running experiements on the model."""

import os
import time
import pickle
import numpy as np

from read_tiff_files import load_data
from model import train_model, predict_model
from sklearn.model_selection  import StratifiedKFold

"""Step 1: Import the data, if its not there load in the tiff files then save as npz for next run."""
try:
    data_pack = np.load("processed_data.npz")
    print("Found Existing Processed Data: ",np.round(os.path.getsize("processed_data.npz")/1e9,3),"GB")
    xdata,ydata,xnames = data_pack['arr_0'], data_pack['arr_1'], data_pack['arr_2']
    del data_pack
    
except:
    print("Did Not Find Existing Data","\n","Importing tiff files")
    xdata, ydata, xnames = load_data()
    print("Saving processed data for faster run next time")
    np.savez("processed_data.npz",xdata, ydata, xnames)
    
patients = []
for name in xnames:
    patient_id  = int(name[:2])
    patients.append(patient_id)
patients = np.asarray(patients)
patients_list = list(set(patients))
patients_status = [np.average(ydata[np.where(patients==patient)[0]]) for patient in patients_list]

#fold returns indexes of 
patient_ix = {patient:np.where(patients==patient)[0] for patient in set(patients)}
timestr=time.strftime("%Y-%m-%d_%H-%M/");os.mkdir(timestr)#create folder for weights, labeled by date-time
log = []
#itterate over left-out patient
for patient in patients_list:
    model_name = timestr+"_patient"+str(patient)

    patient_i = np.where(patients_list==patient)[0][0] #assigned [0-n] value
    train_i = np.where(patients!=patient)[0]
    test_i  = np.where(patients==patient)[0]
    train = [xdata[train_i], ydata[train_i]]
    test = [xdata[test_i], ydata[test_i]]
    
    print("Testing User ", str(patient))
    print("Patient Status ", patients_status[patient_i])
    hist, preds = train_model(train,test,model_name)

    summary = [patient, preds, test[1]]
    log.append(summary)


