import h5py
import datetime
import time
import glob
import os
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def print_pixelmaps(pmaps,id):
        result = (pmaps[id][0])[:, :, 0]
        plt.imshow(result,cmap='Oranges')
        plt.xlim(0,100)
        plt.ylim(0,80)
        plt.show()
        result2 = (pmaps[id][1])[:, :, 0]
        plt.imshow(result2,cmap='Oranges')
        plt.xlim(0,100)
        plt.ylim(0,80)
        
        plt.show()
        return 0  

def splitData(path):
    print("Getting data from: {fn}".format(fn=path))

    filenames = glob.glob(path)
    pm = np.empty((0, 2, 80, 100, 1), dtype=np.uint8)
    lb = np.empty((0), dtype=np.uint8)

    df_list = []
    df_list2 = []

    for (i, fn) in enumerate(filenames):
        print("({ind}/{total}) Loading data from ... {f}".format(f=fn, ind=i + 1, total=len(filenames)))
        cf = h5py.File(fn)

        df_list.append(cf.get('pixelmaps'))
        df_list2.append(cf.get('labels'))
	


    lb = np.concatenate(df_list2)
    pm = np.concatenate(df_list)
    

    print("sorting ...")
    pmaps_train, pmaps_test, y_train, y_test = train_test_split(pm, lb, test_size=1 / 5, random_state=42)
    pmaps_valid, pmaps_test, y_valid, y_test = train_test_split(pmaps_test, y_test, test_size=1 / 2, random_state=42)

    print("saving train set ...")
    dataset = h5py.File("Data_split/data_train.h5", 'w')
    dataset.create_dataset('pixelmaps', data=pmaps_train)
    dataset.create_dataset('labels', data=y_train)
    print("done")

    print("saving test set ...")
    dataset = h5py.File("Data_split/data_test.h5", 'w')
    dataset.create_dataset('pixelmaps', data=pmaps_test)
    dataset.create_dataset('labels', data=y_test)
    print("done")

    print("saving valid set ...")
    dataset = h5py.File("Data_split/data_valid.h5", 'w')
    dataset.create_dataset('pixelmaps', data=pmaps_valid)
    dataset.create_dataset('labels', data=y_valid)
    print("done")

#merge multiple data files into one big file
def mergeData(path):
    print("Getting data from: {fn}".format(fn=path))

    filenames = glob.glob(path)
    pm = np.empty((0, 2, 80, 100, 1), dtype=np.uint8)
    lb = np.empty((0), dtype=np.uint8)

    for (i, fn) in enumerate(filenames):
        print("({ind}/{total}) Loading data from ... {f}".format(f=fn, ind=i + 1, total=len(filenames)))
        cf = h5py.File(fn)
        pm = np.concatenate((pm, cf.get('pixelmaps')), axis=0)
        lb = np.concatenate((lb, cf.get('labels')))

    print("Pixelmaps shape: ", pm.shape)
    print("Labels shape: ", lb.shape)
    print("Number of samples: ", lb.shape[0])

    uq, hist = np.unique(lb, return_counts=True)
    print("Distribution of data: ", hist)

    dataset = h5py.File("DataMerged/dataset_merged.h5", 'w')
    dataset.create_dataset('pixelmaps', data=pm)
    dataset.create_dataset('labels', data=lb)

    print("...Saving merged file of {0} samples...".format(lb.shape))

def get_pmaps(h5file):
    flatmap = h5file.get('rec.training.cvnmaps').get('cvnmap')
    pmaps = np.reshape(flatmap, (-1, 2, 100, 80, 1))
    pmaps = np.transpose(pmaps, (0, 1, 3, 2, 4))
    return pmaps

# Returns a Numpy array of the 5 class labels where:
# numu = 0, nue = 1, nutau = 2, nc = 3, cosmic = 4
def get_labels(h5file):
    pdgs = h5file.get('rec.mc.nu').get('pdg').value
    iscc = h5file.get('rec.mc.nu').get('iscc').value

    mcrun = h5file.get('rec.mc.nu').get('run').value
    mcsubrun = h5file.get('rec.mc.nu').get('subrun').value
    mcevt = h5file.get('rec.mc.nu').get('evt').value
    mcsubevt = h5file.get('rec.mc.nu').get('subevt').value

    prun = h5file.get('rec.training.cvnmaps').get('run').value
    psubrun = h5file.get('rec.training.cvnmaps').get('subrun').value
    pevt = h5file.get('rec.training.cvnmaps').get('evt').value
    psubevt = h5file.get('rec.training.cvnmaps').get('subevt').value

    mcd = { 'run' : np.char.mod('%d', mcrun[:,0]),
            'subrun' : np.char.mod('%d', mcsubrun[:,0]),
            'evt' : np.char.mod('%d', mcevt[:,0]),
            'subevt' : np.char.mod('%d', mcsubevt[:,0]),
            'iscc' : iscc[:,0],
            'pdg' : pdgs[:,0]}

    mcdf = pd.DataFrame(data=mcd)


    pmd = { 'run' : np.char.mod('%d', prun[:,0]),
            'subrun' : np.char.mod('%d', psubrun[:,0]),
            'evt' : np.char.mod('%d', pevt[:,0]),
            'subevt' : np.char.mod('%d', psubevt[:,0]),
            'label' : np.repeat(4, len(prun[:,0]))}

    pmdf = pd.DataFrame(data=pmd)

    zflrun = pmdf.run.map(len).max()
    zflsubrun = pmdf.subrun.map(len).max()
    zflevt = pmdf.evt.map(len).max()
    zflsubevt = pmdf.subevt.map(len).max()

    mcdf['run'] = mcdf['run'].apply(lambda x : x.zfill(zflrun))
    mcdf['subrun'] = mcdf['subrun'].apply(lambda x : x.zfill(zflsubrun))
    mcdf['evt'] = mcdf['evt'].apply(lambda x : x.zfill(zflevt))
    mcdf['subevt'] = mcdf['subevt'].apply(lambda x : x.zfill(zflsubevt))
    mcdf['key'] = mcdf.run+mcdf.subrun+mcdf.evt+mcdf.subevt

    pmdf['run'] = pmdf['run'].apply(lambda x : x.zfill(zflrun))
    pmdf['subrun'] = pmdf['subrun'].apply(lambda x : x.zfill(zflsubrun))
    pmdf['evt'] = pmdf['evt'].apply(lambda x : x.zfill(zflevt))
    pmdf['subevt'] = pmdf['subevt'].apply(lambda x : x.zfill(zflsubevt))
    pmdf['key'] = pmdf.run+pmdf.subrun+pmdf.evt+pmdf.subevt

    nudf = pmdf.loc[pmdf.key.isin(mcdf.key)]
    cosmicdf = pmdf.loc[~pmdf.key.isin(mcdf.key)]

    nudf = pd.merge(nudf, mcdf)

    nudf.loc[abs(nudf.pdg)==12, 'label'] = 1
    nudf.loc[abs(nudf.pdg)==14, 'label'] = 0
    nudf.loc[abs(nudf.pdg)==16, 'label'] = 2
    nudf.loc[nudf.iscc==0, 'label'] = 3

    nudf = nudf.drop(['pdg', 'iscc'], axis=1) # Drop to concat with cosmics

    # Glue the neutrino and cosmic dfs back together
    df = pd.concat([nudf, cosmicdf])

    df = df.sort_values(['key'], ascending=True)
    labels = df['label']

    return np.array(labels)

# Reduce the number of cosmics in the sample to around 10%
def downsample_cosmics(pm, lb):

    """
    If there are more than 10% of cosmics in the dataset,
    then select .1*n indices corresponding to cosmics and
    return those samples.
    """
    ncosmics = np.where(lb==4)[0].shape[0]
    nsamples = lb.shape[0]
    nsel = int(np.floor((0.1*nsamples - 0.1*ncosmics) / 0.9))
    ndel = ncosmics - nsel
    print("%d cosmics out of %d total events and %d will be retained" % (ncosmics, nsamples, nsel))

    # selcosmics / (nsamples-ncosmics)+selcosmics = 0.9
    # selcosmics = 0.9*(nsamples-ncosmics + selcosmics)
    # selcosmics - 0.9*selcosmics = 0.9*nsamples - 0.9*ncosmics
    # selcosmics = (0.9*nsamples - 0.9*ncosmics)/0.1

    if ncosmics <= 0.1*nsamples:
        return pm, lb
    delcosmics = np.sort(random.sample(list(np.where(lb==4)[0]), ndel))
    print(delcosmics)
    print("Downsampling cosmics to 10%...")
    pm = np.delete(pm, delcosmics, axis=0)
    lb = np.delete(lb, delcosmics, axis=0)

    print(pm.shape)
    print(lb.shape)

    return pm, lb


def produce_labeled_h5s(h5dir, samplecosmics=False):
    # The list of files in the directory
    h5files = glob.glob(os.path.join(h5dir, "*.h5"))

    # Create a directory to store the processed files
    # These will persist, so future training can accept these files
    # directly as input
    labeleddir = os.path.join(h5dir, "labeled_downsampled")
    if not os.path.exists(labeleddir):
        os.makedirs(labeleddir)

    i=0
    for filename in h5files:
        print("Processing {fn} at {time}".format(fn=filename, time=datetime.datetime.now()))
        starttime = time.time()
        # Get the current hdf5 file
        currh5 = h5py.File(filename)
        pm = get_pmaps(currh5)
        lb = get_labels(currh5)

        # Reduce the number of cosmics in the final training input
        if samplecosmics:
            pm, lb = downsample_cosmics(pm, lb)

        # Shuffle the dataset here to aid sequential reads later
        shuffle = np.random.permutation(pm.shape[0])
        pm = pm[shuffle, ...]
        lb = lb[shuffle]

        outname = os.path.join(labeleddir, "labeled_{ind}_{fn}".format(ind=i, fn=filename.split('/')[-1]))
        outh5 = h5py.File(outname, 'w')
        outh5.create_dataset('pixelmaps', data=pm)
        outh5.create_dataset('labels', data=lb)
        outh5.close()

        print("{fn} written at {time}".format(fn=outname, time=datetime.datetime.now()))
        print("Elapsed time: {elps} seconds".format(elps=time.time()-starttime))
        print("Events written: {evt}\n".format(evt=pm.shape[0]))
        i += 1

    return labeleddir
