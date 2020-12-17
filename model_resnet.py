import keras
from keras.models import Model
from keras.layers import Activation,Input,Dense,Dropout,BatchNormalization,Conv2D,MaxPooling2D,MaxPooling3D,AveragePooling2D,concatenate,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D,AveragePooling2D
from keras.models import Model
from keras.regularizers import l2

def Unit(x,filters,pool=False):
    res = x
    if pool:
        x = MaxPooling2D(pool_size=(2,2))(x)
        res = Conv2D(filters=filters,kernel_size=[1,1],strides=(2,2),padding="same")(res)
        
    out = BatchNormalization()(x)
    out = Activation("relu")(out)
    out = Conv2D(filters=filters, kernel_size=[3,3], strides=[1, 1], padding="same")(out)

    out = BatchNormalization()(out)
    out = Activation("relu")(out)
    out = Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same")(out)

    out = keras.layers.add([res,out])

    return out

def ResNet():

    images1 = Input(shape=(80,100,1), dtype='float32', name='xview')
    images2 = Input(shape=(80,100,1), dtype='float32', name='yview')
    
    net1=images1
    
    net1 = Conv2D(filters=64, kernel_size=[5, 5], strides=[1, 1], padding="same")(net1)
    
    net1 = Unit(net1,64)
    net1 = Unit(net1,64)
   
    net2 = images2
    
    net2 = Conv2D(filters=64, kernel_size=[5, 5], strides=[1, 1], padding="same")(net2)
    
    net2 = Unit(net2,64)
    net2 = Unit(net2,64)

    net = concatenate([net1, net2])

    net = Unit(net,128,pool=True)
    net = Unit(net,128)
    
    net = Unit(net,256,pool=True)
    net = MaxPooling2D(pool_size=(2,2))(net)
    net = Unit(net,256)
    net = AveragePooling2D(pool_size=(3,3))(net)

    net = Flatten()(net)
    net = Dense(512, activation='relu')(net)
    net = Dense(5,activation='softmax', name="output")(net)

    model = Model(inputs=[images1, images2], outputs=net)


    return model
