from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv3D, Conv2D, MaxPooling2D, MaxPooling3D, GlobalMaxPooling2D, AveragePooling3D
from tensorflow.keras.layers import Activation, Lambda, Reshape
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.layers import GaussianDropout
import tensorflow.keras.backend as K


def build_model(out_type, droprate=0.8, optimizer="Adam", lr=1e-5):
    """this model outputs overlapping windows. 
    The span is 40 px between windows. Each window is 160x160. 
    Windows are only in x, y, the entire depth is used. 
    """
    inputs = Input(shape=(10, 800, 800, 4))
    x = Lambda(lambda x: x/4095.)(inputs)
    x = GaussianDropout(0.15)(x)
    
    x = Conv3D(16, (1,2,2), strides=(1,2,2), kernel_initializer='lecun_normal')(x)
    #x = AveragePooling3D((1,2,2))(x)
    x = Activation("selu")(x)#400,400
    
    x = Conv3D(16, (2,3,3), kernel_initializer='lecun_normal')(x)#(398,398)
    x = Activation("selu")(x)
    x = Conv3D(16, (3,3,3), dilation_rate=(2,2,2), kernel_initializer='lecun_normal')(x)#(394,394)
    x = AveragePooling3D((5,2,2))(x)#(197,197)
    x = Activation("selu")(x)
    x = Reshape((197,197,16))(x)#reduce from 3D to 2D

    x = Conv2D(16, (2,2), kernel_initializer='lecun_normal')(x)#(196,196)
    x = Activation("selu")(x)
    x = Conv2D(32, (3,3), dilation_rate=(2,2), kernel_initializer='lecun_normal')(x)#(192,192)
    x = MaxPooling2D((2, 2))(x)#(96,96)
    x = Activation("selu")(x)

    x = Conv2D(32, (3,3), kernel_initializer='lecun_normal')(x)#(94,94)
    x = Activation("selu")(x)
    x = Conv2D(64, (3,3), dilation_rate=(2,2), kernel_initializer='lecun_normal')(x)#(90,90)
    x = MaxPooling2D((5, 5))(x)#(18,18)
    x = Activation("selu")(x)
    
    x = Conv2D(64, (2,2), kernel_initializer='lecun_normal')(x)#(17,17)
    x = Activation("selu")(x)

    #rc:160@40
    x = Conv2D(32, 1, activation="selu", kernel_initializer="lecun_normal")(x)
    x = Conv2D(8,  1, activation="selu", kernel_initializer="lecun_normal")(x)
    heat_map = Conv2D(1,  1, activation="sigmoid")(x)
    
    UnscaledDrop = Lambda(lambda x : K.in_train_phase((1.-droprate)*K.dropout(x, droprate), x))
    drop_map = UnscaledDrop(heat_map)
    
    prob = GlobalMaxPooling2D()(drop_map)
    
    if out_type=='heat_map':y=heat_map
    if out_type=='class':y=prob
    model = Model(inputs=inputs, outputs=y)
    
    if optimizer=="Nadam":opt = Nadam(lr=lr)
    if optimizer=="Adam":opt = Adam(lr=lr)
    model.compile(loss="binary_crossentropy",optimizer=opt, metrics=['accuracy'])

    return model


"""
Convolution	Pool/Stride	Size	RF	Stride
1	1	800	1	1
2	2	400	2	2
3	1	398	6	2
5	1	394	14	2
2	2	197	16	4
2	1	196	20	4
5	1	192	36	4
2	2	96	40	8
3	1	94	56	8
5	1	90	88	8
5	5	18	120	40
2	1	17	160	40

"""
