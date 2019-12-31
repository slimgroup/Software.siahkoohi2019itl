import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--hdf5path', dest='hdf5path', type=str, default='/home/ec2-user/model', help='path')
parser.add_argument('--save_dir', dest='save_dir', type=str, default='default', help='saving directory')
parser.add_argument('--shotNum', dest='shotNum', type=int, default=215, help='shot number to plot')

args = parser.parse_args()

hdf5path  = args.hdf5path
hdf5name = os.path.join(hdf5path, 'mapping_result.hdf5')
save_dir = args.save_dir
shotNum = ags.shotNum

strName = hdf5name
dataset_name = "result"

fileName = h5py.File(strName, 'r')
data_num = fileName[dataset_name].shape[0]
data_numA = fileName[dataset_name + 'A'].shape[0]
data_numB = fileName[dataset_name + 'B'].shape[0]

data_train = fileName[dataset_name][shotNum,:,:]
data_trainA = fileName[dataset_name + 'A'][shotNum,:,:]
data_trainB = fileName[dataset_name + 'B'][shotNum,:,:]

dt = 12 * 0.278 * 1e-3
dx = 5.59375
Xstart= (data_trainA.shape[1]-shotNum) * dx
X0 = (-shotNum + 1) * dx
Tend= data_trainA.shape[0]*dt


font = {'family' : 'sans-serif',
        'size'   : 16}
import matplotlib
matplotlib.rc('font', **font)

plt.figure(); im = plt.imshow(data_trainA*150, vmin=-1, vmax=1, cmap="Greys", aspect='auto', 
	extent=[X0,Xstart,Tend,0], interpolation="lanczos")
plt.xlabel('Offset (m)')
plt.ylabel('Time (s)')
plt.colorbar(im,fraction=0.038, pad=0.02)
plt.savefig(os.path.join(save_dir, 'ModeledWithoutSRM.eps'), format='eps', bbox_inches='tight', dpi=600)

plt.figure(); im = plt.imshow(data_train*150, vmin=-1, vmax=1, cmap="Greys", aspect='auto', 
	extent=[X0,Xstart,Tend,0], interpolation="lanczos")
plt.xlabel('Offset (m)')
plt.ylabel('Time (s)')
plt.colorbar(im,fraction=0.038, pad=0.02)
plt.savefig(os.path.join(save_dir, 'SRME_result.eps'), format='eps', bbox_inches='tight', dpi=600)

plt.figure(); im = plt.imshow(data_trainB*150, vmin=-2, vmax=2, cmap="Greys", aspect='auto', 
	extent=[X0,Xstart,Tend,0], interpolation="lanczos")
plt.xlabel('Offset (m)')
plt.ylabel('Time (s)')
plt.colorbar(im,fraction=0.038, pad=0.02)
plt.savefig(os.path.join(save_dir, 'ModeledWithSRM.eps'), format='eps', bbox_inches='tight', dpi=600)

plt.figure(); im = plt.imshow(data_trainA*150-data_train*150, vmin=-1, vmax=1, cmap="Greys", aspect='auto', 
	extent=[X0,Xstart,Tend,0], interpolation="lanczos")
plt.xlabel('Offset (m)')
plt.ylabel('Time (s)')
plt.colorbar(im,fraction=0.038, pad=0.02)
plt.savefig(os.path.join(save_dir, 'diff.eps'), format='eps', bbox_inches='tight', dpi=600)

