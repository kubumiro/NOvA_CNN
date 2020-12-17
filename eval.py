import h5py
import datetime
import time
import glob
import os
import random
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.utils import np_utils
from sklearn import metrics
from matplotlib import pyplot as plt
import h5py
import itertools

def plot_loss(name):

    path = 'Models/history_{0}.csv'.format(name)
    history = pd.read_csv(path)
    epochs_trained = len(history['loss'])

    plt.plot(range(1,epochs_trained+1),history['loss'],color = "green",  label='Loss Train')
    plt.plot(range(1,epochs_trained+1), history['val_loss'], color = "blue",label='Loss Valid')
    plt.xlabel('Epoch',fontsize=13)
    plt.ylabel('Cross-Entropy Loss',fontsize=13)
    plt.legend(fontsize=13)
    plt.savefig('Plots/loss_{0}.png'.format(name))
    plt.clf()

    plt.plot(range(1,epochs_trained+1),history['acc'],color = "green",  label='ACC Train')
    plt.plot(range(1,epochs_trained+1), history['val_acc'], color = "blue",label='ACC Valid')
    plt.xlabel('Epoch',fontsize=13)
    plt.ylabel('Cross-Entropy Loss',fontsize=13)
    plt.legend(fontsize=13)
    plt.savefig('Plots/acc_{0}.png'.format(name))
    plt.clf()


def plot_confusion_matrix(cm, classes, path, name, normalize=False,cmap=plt.cm.Blues):
    from matplotlib import rcParams
    rcParams.update({'figure.autolayout': True})
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        digit = '.4f'
        path = path + '_norm'
     
    else:
        print('Confusion matrix, without normalization')
        digit = '.0f' 

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.colorbar()
    if normalize:
        plt.clim(0,1)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], digit),
                 horizontalalignment="center", fontsize = 13,
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label',fontsize=13)
    plt.xlabel('Predicted label',fontsize=13)
    plt.savefig('Plots/{m}/CM_{m}_{p}.png'.format(m=name,p=path))

    plt.clf()


def ControlPlots(name, path):    
    print("..... Plotting histograms for {m} on {p}".format(m=name,p=str(path[-12:-3])))
    hf = h5py.File('Predictions/{m}_{p}.h5'.format(m=name,p=str(path[-12:-3])), 'r')
    y_pred = np.array(hf.get('y_pred'))
    y = np.array(hf.get('y_true'))

    col = ['r','g','b','magenta','cyan']
    class_names=[r'$\nu_{\mu}$',r'$\nu_e$',r'$\nu_{\tau}$','NC','cosmics']
    
    Classes = []
    for i in range(0, 5):
        Classes.append(y_pred[np.argmax(y,axis=1)==i])
    
    for i,Class in enumerate(Classes):
        print("... plotting histogram for {0} ...".format(class_names[i]))
        for j in range(0, 5):
             plt.hist(Class[:,j],alpha = 0.7, bins = 'doane', color = col[j], label = class_names[j], histtype=u'step', density = True, linewidth = 2)
             plt.legend(fontsize = 16)
             plt.xlim([0-0.005,1+0.005])
             plt.xlabel("Discriminant", fontsize = 16)
             plt.ylabel("Normalized frequency of events", fontsize = 16)
             plt.tick_params(axis='both', labelsize=16)

        plt.savefig('Plots/{m}/CP2_{m}_{p}_{c}.png'.format(m=name,p=path[-12:-3], c = i))
        plt.clf()



def Evaluate(name, path, histo = False):
  
    print("... evaluating model {0} ...".format(name))
    hf = h5py.File('Predictions/{m}_{p}.h5'.format(m=name,p=str(path[-12:-3])), 'r')
    y_pred = np.array(hf.get('y_pred'))
    y = np.array(hf.get('y_true'))


    print("acc ... ",metrics.accuracy_score(np.argmax(y,axis=1), np.argmax(y_pred,axis=1)))

    print("... Calculating confusion matrix ...")
    cnf_matrix = metrics.confusion_matrix(np.argmax(y,axis=1), np.argmax(y_pred,axis=1))
    np.set_printoptions(precision=2)


    class_names=[r'$\nu_{\mu}$',r'$\nu_e$',r'$\nu_{\tau}$','NC','cosmics']

    plot_confusion_matrix(cnf_matrix, classes=class_names, path = str(path[-12:-3]), name=name)
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,  path = str(path[-12:-3]), name=name)
    
    with open("eval.txt", "a") as text_file:
        text_file.write("{m} on {p}\n".format(m = name, p = path))
        text_file.write("confusion matrix:\n {0} \n".format(metrics.confusion_matrix(np.argmax(y,axis=1), np.argmax(y_pred,axis=1))))
        text_file.write("metrics:\n {0} \n".format(metrics.classification_report(np.argmax(y,axis=1), np.argmax(y_pred,axis=1), digits=3)))
    print("Evaluation done.")
    try:
        plot_loss(name)
    except:
        print("Done.")
    if histo:
        ControlPlots(name,path)

def GetData(path):
    print("Getting data from: {fn}".format(fn=path))

    filenames = glob.glob(path)
    pm = np.empty((0, 2, 80, 100, 1), dtype=np.uint8)
    lb = np.empty((0), dtype=np.uint8)

    for (i,fn) in enumerate(filenames):
        print("({ind}/{total}) Loading data from ... {f}".format(f=fn,ind = i+1,total = len(filenames)))
        cf = h5py.File(fn)
        pm = np.concatenate((pm, cf.get('pixelmaps')), axis=0)  #0
        lb = np.concatenate((lb, cf.get('labels')))

    print("Pixelmaps shape: ",pm.shape)
    print("Labels shape: ",lb.shape)
    print("Number of samples: ",lb.shape[0])

    uq, hist = np.unique(lb, return_counts=True)
    print("Distribution of data: ",hist) 
    return pm,lb


def GetTrainingSet(path):

    pm, lb = GetData(path)
    number_of_classes = 5

    
    X1_train = pm[:, 0]
    X2_train = pm[:, 1]
    
    #X1_train = X1_train.astype('float32') / 255
    #X2_train = X2_train.astype('float32') / 255
    
    y = np_utils.to_categorical(lb, number_of_classes)

    return X1_train, X2_train, y

