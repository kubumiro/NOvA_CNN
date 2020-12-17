import keras
from keras.layers import (Input, Conv2D, Flatten, Dense, Dropout,
                          MaxPool2D, concatenate, AveragePooling2D,
                          BatchNormalization)
from keras.models import Model
from keras.regularizers import l2

# default layer order seems to change from version 2.1.6 to 2.2 which I've tested on locally
if keras.__version__ == '2.1.6':
    concataxis = 3
else:
    concataxis = 3
# Inception module function. Filters correspond to those in GoogLeNet
# mode='sub' refers to an Inception call from the subnet
# mode='comb' refers to an Inception call from the combined network
def Inception(input_layer, mode='sub'):
    # Filters for the subnet Inception pass
    if mode == 'sub':
        filters = { 'c1x1' : 64, 'c3x3a' : 96, 'c3x3b' : 128,
                'c5x5a' : 16, 'c5x5b' : 32, 'pool' : 32 }
    # Filters for the combined Inception pass
    elif mode == 'comb':
        filters = { 'c1x1' : 384, 'c3x3a' : 192, 'c3x3b' : 384,
                'c5x5a' : 48, 'c5x5b' : 128, 'pool' : 128 }
    else:
        raise ValueError("Inception module mode should be either sub or comb.")

    conv1x1 = Conv2D(filters['c1x1'],
                     (1,1),
                     activation='relu',
                     kernel_regularizer=l2(0.0002))(input_layer)
    conv1x1 = BatchNormalization()(conv1x1)
    conv3x3 = Conv2D(filters['c3x3a'],
                     (1,1),
                     activation='relu',
                     kernel_regularizer=l2(0.0002))(input_layer)
    conv3x3 = BatchNormalization()(conv3x3)
    conv3x3 = Conv2D(filters['c3x3b'],
                     (3,3),
                     activation='relu',
                     padding='same',
                     kernel_regularizer=l2(0.0002))(conv3x3)
    conv3x3 = BatchNormalization()(conv3x3)

    conv5x5 = Conv2D(filters['c5x5a'],
                     (1,1),
                     activation='relu',
                     kernel_regularizer=l2(0.0002))(input_layer)
    conv5x5 = BatchNormalization()(conv5x5)
    conv5x5 = Conv2D(filters['c5x5b'],
                     (5,5),
                     activation='relu',
                     padding='same',
                     kernel_regularizer=l2(0.0002))(conv5x5)
    conv5x5 = BatchNormalization()(conv5x5)

    pool = MaxPool2D(pool_size=(3, 3), strides=1, padding='same')(input_layer)
    pool = Conv2D(filters['pool'],
                  (1,1),
                  activation='relu',
                  kernel_regularizer=l2(0.0002))(input_layer)
    pool = BatchNormalization()(pool)


    output = concatenate([conv1x1, conv3x3, conv5x5, pool], axis=concataxis)

    return output

# The subnetwork that each view passes through in parallel
def Subnet(layer):
    layer = Conv2D(64,
                 (7,7),
                 activation='relu',
                 padding='same',
                 strides=2,
                 kernel_regularizer=l2(0.0002))(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPool2D(pool_size=(3, 3),
                    strides=2,
                    padding='valid',
                    data_format='channels_last')(layer)
    layer = Conv2D(64,
                 (1,1),
                 activation='relu',
                 kernel_regularizer=l2(0.0002))(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(192,
                 (3,3),
                 activation='relu',
                 padding='same',
                 strides=1,
                 kernel_regularizer=l2(0.0002))(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPool2D(pool_size=(3, 3),
                    strides=2,
                    data_format='channels_last')(layer)
    layer = Inception(layer, mode='sub')
    output = MaxPool2D(pool_size=(3, 3),
                     strides=2,
                     data_format='channels_last')(layer)

    return output

# Model defintion for CVNShortSimple
def CVNShortSimple():
    xview = Input(shape=(80, 100, 1), dtype='float32', name='xview')
    yview = Input(shape=(80, 100, 1), dtype='float32', name='yview')

    xoutput = Subnet(xview)
    youtput = Subnet(yview)

    merged = concatenate([xoutput,youtput], axis=concataxis)
    merged = Inception(merged, mode='comb')

    merged = AveragePooling2D(pool_size=(6, 5), padding='same', strides=1)(merged)
    merged = Flatten()(merged)
    merged = Dropout(0.4)(merged)

    output = Dense(units=5,
                 activation='softmax',
                 name='output')(merged)

    model = Model(inputs=[xview, yview], outputs=[output])

    return model



# Model defintion for CVNShortSimple
def CVNShortSimple_Dense():
    xview = Input(shape=(80, 100, 1), dtype='float32', name='xview')
    yview = Input(shape=(80, 100, 1), dtype='float32', name='yview')

    xoutput = Subnet(xview)
    youtput = Subnet(yview)

    merged = concatenate([xoutput,youtput], axis=concataxis)
    merged = Inception(merged, mode='comb')

    merged = AveragePooling2D(pool_size=(6, 5), padding='same', strides=1)(merged)
    merged = Flatten()(merged)
    merged = Dropout(0.4)(merged)
    merged = Dense(units = 128, activation = 'relu')(merged)

    output = Dense(units=5,
                 activation='softmax',
                 name='output')(merged)

    model = Model(inputs=[xview, yview], outputs=[output])

    return model
