from keras import Model, Input
from keras.layers import Conv3D, Conv2D, MaxPooling2D, MaxPooling3D, GlobalMaxPooling2D, AveragePooling3D, AveragePooling2D
from keras.layers import Activation, Lambda, Reshape, BatchNormalization
from keras.optimizers import SGD, Adam, Nadam
from keras.layers import GaussianDropout, GaussianNoise
from keras.initializers import Constant
from model_functions import accumulate_gradient, tptn_metric
import keras.backend as K


def build_model(out_type, sizes, droprate=0.8, optimizer="Adam", lr=1e-5, grad_accum=1):
    """this model outputs overlapping windows. 
    The span is 40 px between windows. Each window is 160x160. 
    Windows are only in x, y, the entire depth is used. 
    
    This model is vaguely styled after VGG, but special care is taken to ensure the
    specific receptive field and stride we were looking for. This means that the network isn't
    really made up of repeating [conv-3, conv-3, pool-2] blocks. 
    
    For those learning how to work with receptive field, I suggest you study this:
        https://arxiv.org/abs/1603.07285
    Personally, I have those equations implimented into an excel sheet so I can just
    enter in kernel_size and "stride/pool size" columns to tweak things until I get the right parameters
    """
    s = sizes
    inputs = Input(shape=(10, 800, 800, 4))
    x = Lambda(lambda x: x/4095.)(inputs)
    
    #brightness shift: uniform
    x = Lambda(lambda x: K.in_train_phase(x*K.random_normal((1,1,1,1,1), 1.0, 0.15), x))(x)#additional augmentation. This is here rather than in the generator because we want to keep slimmer int data and do expensive float ops on gpu
    #x = GaussianNoise(0.15)(x)
    x = Conv3D(s[0], (1,3,3), activation="selu", kernel_initializer='lecun_normal')(x)#10,798,798
    x = Conv3D(s[0], (3,3,3), dilation_rate=(1,2,2), activation="selu", kernel_initializer='lecun_normal')(x)#8,796,796
    x = Conv3D(s[0], (3,3,3), dilation_rate=(2,2,2), activation="linear", kernel_initializer='lecun_normal')(x)#4,796,796
    x = MaxPooling3D((4,5,5))(x)#1,158,158; This maximizes over the entire depth, less variance on padded depth dim. After this, we use avg pooling because gradients are already pretty sparse from the globalmax at the end
    x = Reshape((158, 158, s[0]))(x)#158; we are now 2D
    x = Activation("selu")(x)

    
    x = Conv2D(s[1], 1, activation="selu", kernel_initializer='lecun_normal')(x)#158#these 1x1 conv are basically pixel-distributed dense layer processing on the channels
    x = Conv2D(s[1], 3, activation="linear", kernel_initializer='lecun_normal')(x)#156
    x = MaxPooling2D(2)(x)#78
    x = Activation("selu")(x)


    x = Conv2D(s[2], 3, activation="selu", kernel_initializer='lecun_normal')(x)#76
    x = Conv2D(s[2], 3, activation="linear", kernel_initializer='lecun_normal')(x)#74
    x = MaxPooling2D(2)(x)#37
    x = Activation("selu")(x)

    
    x = Conv2D(s[3], 3, activation="selu", kernel_initializer='lecun_normal')(x)#35
    x = Conv2D(s[3], 1, activation="selu", kernel_initializer='lecun_normal')(x)#17
    x = MaxPooling2D(3, strides=2)(x)
    
    #rc:160@40 
    x = Conv2D(s[4], 1, activation="selu", kernel_initializer='lecun_normal')(x)
    heat_map = Conv2D(1, 1, activation="sigmoid")(x)
    
    UnscaledDrop = Lambda(lambda x : K.in_train_phase((1.-droprate)*K.dropout(x, droprate, noise_shape=(1,17,17,1)), x))#we just define the dropout here, we just want to cancel out the scaling that's a normal part of dropout
    drop_map = UnscaledDrop(heat_map)#we apply the dropout function defined above
    
    prob = GlobalMaxPooling2D()(drop_map)#our patch "fusion"
    
    if out_type=='heat_map':y=heat_map
    if out_type=='class':y=prob
    model = Model(inputs=inputs, outputs=y)
    
    if optimizer=="SGD":opt_ = SGD(lr=lr, momentum=0.9)
    if optimizer=="Adam":opt_ = Adam(lr=lr)
    if optimizer=="Nadam":opt_ = Nadam(lr=lr)
    
    if grad_accum>1:opt = accumulate_gradient(opt_, grad_accum)
    if grad_accum==1:opt = opt_
    
    model.compile(loss="binary_crossentropy",optimizer=opt, metrics=['accuracy'])

    return model

