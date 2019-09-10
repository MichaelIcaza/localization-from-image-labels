#Begin Keras
import keras.backend as K
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.layers import Input, Activation
from keras.layers.core import Dense, Dropout, Reshape, Lambda, Flatten
from keras.layers.convolutional import Conv3D, Conv2D, UpSampling2D
from keras.layers.merge import average, concatenate, add
from keras.layers.noise import GaussianNoise as Noise
from keras.layers.noise import AlphaDropout
from keras.layers.normalization import BatchNormalization as BN
from keras.layers.pooling import GlobalMaxPooling3D, MaxPooling3D, AveragePooling3D, MaxPooling2D, GlobalMaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.models import Model
from keras.optimizers import RMSprop, Nadam, Adam
from keras.preprocessing.image import ImageDataGenerator as Generator

import numpy as np
import tensorflow as tf

np.random.seed = 123
tf.set_random_seed(123)
def augment_image(img):#manipulates (height, width, depth, ch) images.
    img_aug = np.rot90(img, k=np.random.randint(0,3), axes=(1,2)) #90-degree rotation
    if np.random.rand()>0.5:img_aug=np.transpose(img_aug,axes=(0,2,1,3))
    img_aug *= np.random.normal(1.0,0.15)
    return img_aug
    
def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def Generator(data, shuffle, augment):
    while True: #The generator must run continiously with a yield rather than return
        xsamples, ysamples = data[0], data[1]
        index = np.arange(xsamples.shape[0]) #1- build [0, n-1] index
        
        pos = np.sum(ysamples)#class weights
        neg = len(ysamples)-pos
        class_weight = {0:pos/neg, 1:neg/pos}#goes to "Sample_weights" below
        if pos==0 or neg==0:class_weight={0:1,1:1}
        if shuffle:np.random.shuffle(index) #2- Shuffle Index
        for i in index:
            xout = xsamples[i:i+1]/4095. #normalize the values to [0,1]
            yout = ysamples[i:i+1]
            if augment:xout=np.asarray([augment_image(img) for img in xout])#random augmentations to prevent overfitting
            sample_weight = np.asarray([class_weight[y] for y in yout])
            yield xout, yout, sample_weight

def build_model(out_type):
    """this model outputs overlapping windows. 
    The span is 40 px between windows. Each window is 160x160. 
    Windows are only in x, y, the entire depth is used. 
    """
    inputs = Input(shape=(10, 400, 400, 4))
    #x = Noise(0.1)(inputs)#shrink moved to preprocessing
    x = Conv3D(16, (2,3,3), kernel_initializer='lecun_normal')(inputs)#(398,398)
    x = Activation("selu")(x)
    x = Conv3D(16, (3,3,3), dilation_rate=(2,2,2), kernel_initializer='lecun_normal')(x)#(394,394)
    x = MaxPooling3D((5,2,2))(x)#(197,197)
    x = Activation("selu")(x)
    x = Reshape((197,197,-1))(x)

    x = Conv2D(16, (2,2), kernel_initializer='lecun_normal')(x)#(196,196)
    x = Activation("selu")(x)
    x = Conv2D(16, (3,3), dilation_rate=(2,2), kernel_initializer='lecun_normal')(x)#(192,192)
    x = MaxPooling2D((2, 2))(x)#(96,96)
    x = Activation("selu")(x)

    x = Conv2D(32, (3,3), kernel_initializer='lecun_normal')(x)#(94,94)
    x = Activation("selu")(x)
    x = Conv2D(32, (3,3), dilation_rate=(2,2), kernel_initializer='lecun_normal')(x)#(90,90)
    x = MaxPooling2D((5, 5))(x)#(18,18)
    x = Activation("selu")(x)
    
    x = Conv2D(64, (2,2), kernel_initializer='lecun_normal')(x)#(17,17)
    x = Activation("selu")(x)

    #rc:160@40
    x = Conv2D(32, (1,1))(x)
    x = Activation("relu")(x)
    x = Conv2D(16, (1,1))(x)
    x = Activation("relu")(x)
    x = Conv2D(1, (1,1))(x)
    heat_map = Activation("sigmoid")(x)
    
    UnscaledDrop = Lambda(lambda x : K.in_train_phase((0.5)*K.dropout(x, 0.5), x))
    drop_map = UnscaledDrop(heat_map)
    
    prob = GlobalMaxPooling2D()(drop_map)
    
    if out_type=='heat_map':y=heat_map
    if out_type=='class':y=prob
    model = Model(inputs=inputs, outputs=y)

    return model

def train_model(train,test,weights_name):
    train_size, val_size = len(train[0]), len(test[0])
    
    checkpt = ModelCheckpoint(weights_name+".hdf5", monitor="loss", verbose=0, save_best_only=True)
    earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=0)
    rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=1e-6)
    
    model = build_model(out_type="class")
    model.compile(loss="binary_crossentropy",optimizer=Adam(lr=0.0001), metrics=['accuracy'])
    hist = model.fit_generator(Generator(train,True,True),validation_data=Generator(test,False,False),
                        epochs=100, steps_per_epoch=train_size, validation_steps=val_size, verbose=1,
                        max_queue_size=8,callbacks=[checkpt, earlystop])
    model.load_weights(weights_name+".hdf5")
    preds = model.predict_generator(Generator(test,False,False), steps=val_size)
    
    return hist, preds


def predict_model(test,weights_name,verbose):
    print("Creating New Model")
    val_size = len(test[0])   
    model = build_model(out_type="class")
    model.load_weights(weights_name+".hdf5")
    model.compile(loss="binary_crossentropy",optimizer=RMSprop(lr=2e-4), metrics=['accuracy'])
    preds = model.predict_generator(Generator(test, False, False), steps=val_size, max_queue_size=10,verbose=verbose)
    return preds


def evaluate_model(x, y, weights_name):
    print("Running Evaluation on:")
    print("\t"+weights_name)
    model = build_model(out_type='class')
    model.load_weights(weights_name)
    model.compile(loss="binary_crossentropy",optimizer=Nadam(), metrics=['accuracy'])
    preds = model.predict_generator(Generator((x,y),False,False), steps=len(x))
    return preds

def parallel_shuffle(x, y, names): #len(x) must = len(y)
    index = np.arange(x.shape[0]) #1- build [0, n-1] index
    np.random.shuffle(index) #2- Shuffle Index
    x = x[index] #Python Magic
    y = y[index]
    names = names[index]
    return x, y, names