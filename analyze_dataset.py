import os
import glob
import h5py
import numpy as np

filenames = glob.glob('/data/gniksere/largesubset/test_lb_ds_bal/*.h5')

pm = np.empty((0, 2, 80, 100, 1), dtype=np.uint8)
lb = np.empty((0), dtype=np.uint8)

for fn in filenames:
    cf = h5py.File(fn)
    pm = np.concatenate((pm, cf.get('pixelmaps')), axis=0)
    lb = np.concatenate((lb, cf.get('labels')))

print(pm.shape)
print(pm)
print(lb.shape)

uq, hist = np.unique(lb, return_counts=True)

print(uq)
print(hist) 
