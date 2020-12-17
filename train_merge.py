import numpy as np

import keras
from keras.utils import plot_model
from keras.optimizers import SGD

from model import CVNShortSimple
from datagenerator import DataGenerator
from generator_merge import generator
from args import parse_args
from hdf5tools import produce_labeled_h5s

def TrainCVNShortSimple(args):
    traindir = args.traindir
    testdir = args.testdir

    if args.label:
        traindir = produce_labeled_h5s(args.traindir, args.reducecosmics)
        testdir = produce_labeled_h5s(args.testdir, args.reducecosmics)

    # train_batch = DataGenerator(traindir, args.trbatchsize)
    # test_batch = DataGenerator(testdir, args.tsbatchsize)
    train_batch = generator(traindir, args.trbatchsize)
    test_batch = generator(testdir, args.tsbatchsize)

    model = CVNShortSimple()
    optimizer = SGD(lr=args.baselr, momentum=args.momentum)
    model.compile(optimizer=optimizer,
                  loss={'output':'categorical_crossentropy'},
                  loss_weights={'output' : 1.0},
                  metrics=['accuracy'])
    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/out_weight_{}'.format(1.0))
    history   = model.fit_generator(generator=train_batch, steps_per_epoch=args.ntrsteps,
                                    epochs=args.nepochs, verbose=1,
                                    callbacks=[tensorboard],
                                    validation_data=test_batch, validation_steps=args.ntssteps)
#if __name__ == "__main__":
#    args = parse_args()
#    TrainCVNShortSimple(args)
