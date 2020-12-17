import keras
from keras.models import Model
from keras.layers import Activation,Input,Dense,Dropout,BatchNormalization,Conv2D,MaxPooling2D,MaxPooling3D,AveragePooling2D,concatenate,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D,AveragePooling2D
from keras.models import Model
from keras.regularizers import l2

def ShortCNN():

    images1 = Input(shape=(80,100,1), dtype='float32', name='xview')
    images2 = Input(shape=(80,100,1), dtype='float32', name='yview')
    
    net1=images1
    net1 = Conv2D(filters=32, kernel_size=[5, 5], strides=[1, 1], padding="same")(net1)
    net1 = AveragePooling2D(pool_size=(2,2))(net1)

    net1 = BatchNormalization()(net1)

    net2 = images2  
    net2 = Conv2D(filters=32, kernel_size=[5, 5], strides=[1, 1], padding="same")(net2)
    net2 = AveragePooling2D(pool_size=(2,2))(net2)
    net2 = BatchNormalization()(net2)

    net = concatenate([net1, net2])
    net = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding="same")(net)
    net = AveragePooling2D(pool_size=(4,4))(net)

    net = Flatten()(net)
    net = Dense(128, activation='relu')(net)
    net = Dense(5,activation='softmax', name="output")(net)

    model = Model(inputs=[images1, images2], outputs=net)


    return model