import numpy as np
import pandas as pd
import keras
import h5py
from keras.utils import plot_model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from model import CVNShortSimple, CVNShortSimple_Dense
from model_resnet import ResNet
from model_cnn import ShortCNN
from generator import generator
from args import parse_args
from hdf5tools import produce_labeled_h5s
from eval import plot_confusion_matrix, GetTrainingSet, plot_loss
from dask.distributed import Client, progress
import dask.array as da

class ModelCNN:

    def __init__(self, name):
        self.name = name
        self.args = parse_args()

    def Train(self):
        args = self.args
        traindir = args.traindir
        validdir = args.validdir

    # Preprocess the labels if the -l flag is passed to train.py
    #if args.label:
    #    traindir = produce_labeled_h5s(args.traindir, args.reducecosmics)
    #    testdir = produce_labeled_h5s(args.testdir, args.reducecosmics)

    # The generator objects for the train and test sets
        train_batch = generator(traindir, args.trbatchsize)
        test_batch = generator(validdir, args.tsbatchsize)

    # Initialize the model, optimizer, and callbacks then train

        if self.name == "CVN":
            model = CVNShortSimple()
        elif self.name == "CVN_Dense":
            model = CVNShortSimple_Dense()          
        elif self.name == "ResNet":
            model = ResNet()
        elif self.name == "ShortCNN":
            model = ShortCNN()

        print(model.summary())

        optimizer = SGD(lr=args.baselr, momentum=args.momentum)
        model.compile(optimizer=optimizer,
                  loss={'output':'categorical_crossentropy'},
                  loss_weights={'output' : 1.0},
                  metrics=['accuracy'])
        tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/out_weight_{}'.format(1.0))
        history   = model.fit_generator(generator=train_batch, steps_per_epoch=args.ntrsteps,
                                    epochs=args.nepochs, verbose=1, callbacks=[tensorboard, ModelCheckpoint('Models/model_{0}_weights.h5'.format(self.name), save_best_only=True),
                                    EarlyStopping(patience=10)],
                                    validation_data=test_batch, validation_steps=args.ntssteps)



    # update the model with the best weights
        model.load_weights('Models/model_{0}_weights.h5'.format(self.name))

    # save the best model
        print("... saving model {0} ...".format(self.name))
        model.save('Models/{0}.h5'.format(self.name))

        print("... saving model history ...".format(self.name))
        hist_df = pd.DataFrame(history.history) 
        hist_csv_file = 'Models/history_{0}.csv'.format(self.name)
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)


    def Predict(self,path, large = False):    

        path_model = 'Models/{0}.h5'.format(self.name)
        print("... loading model from ... ", path_model)
        model = load_model(path_model)
        if large == False:
            print("... gettint evaluation dataset from ... ", path)
            X1, X2, y = GetTrainingSet(path)
            print("... calculating predictions ... ")  
            y_pred = model.predict([X1,X2], verbose=1)
        elif large == True:
            print("... gettint evaluation dataset from ... ", path)
            X1, X2, y = GetTrainingSet(path)
            print("... calculating predictions ... ")  
            y_pred = model.predict([X1,X2], verbose=1)


        print("... saving predictions ... ")
        hf = h5py.File('Predictions/{m}_{p}.h5'.format(m=self.name,p=str(path[-12:-3])), 'w')
        hf.create_dataset('y_pred', data=y_pred)
        hf.create_dataset('y_true', data=y)
        hf.close()





