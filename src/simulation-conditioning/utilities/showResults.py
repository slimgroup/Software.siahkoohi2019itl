import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--hdf5path', dest='hdf5path', type=str, default='/home/ec2-user/model', help='path')
parser.add_argument('--save_dir', dest='save_dir', type=str, default='default', help='saving directory')
parser.add_argument('--testNum', dest='testNum', type=int, default=6392, help='wavefield snapshot index to plot')

args = parser.parse_args()

hdf5path  = args.hdf5path
hdf5name = os.path.join(hdf5path, 'mapping_result.hdf5')
save_dir = args.save_dir
testNum = ags.testNum

strName = hdf5name
dataset_name = "result"

fileName = h5py.File(strName, 'r')
data_num = fileName[dataset_name].shape[0]
data_numA = fileName[dataset_name + 'A'].shape[0]
data_numB = fileName[dataset_name + 'B'].shape[0]

data_train = fileName[dataset_name][testNum,:,:]
data_trainA = fileName[dataset_name + 'A'][testNum,:,:]
data_trainB = fileName[dataset_name + 'B'][testNum,:,:]

dt = 8 * 0.675 * 1e-3
dx = 7.5
Xstart= data_trainA.shape[0] * dx
Tend= data_trainA.shape[1]*dx

font = {'family' : 'sans-serif',
        'size'   : 16}
import matplotlib
matplotlib.rc('font', **font)

plt.figure()
im = plt.imshow(np.transpose(data_trainA), vmin=-.5, vmax=.5, cmap="Greys", aspect='1', \
	extent=[0,Xstart,Tend,0], interpolation="lanczos")
plt.xlabel('Horizontal Location (m)')
plt.ylabel('Depth (m)')
plt.colorbar(im,fraction=0.034, pad=0.06)
plt.grid(linestyle='--', linewidth=1, alpha=1, color='k')
plt.savefig(os.path.join(save_dir, 'wave-NonDispersed.eps'), format='eps', bbox_inches='tight', dpi=300)

plt.figure()
im = plt.imshow(np.transpose(data_train),  vmin=-.5, vmax=.5, cmap="Greys", aspect='1', \
	extent=[0,Xstart,Tend,0], interpolation="lanczos")
plt.xlabel('Horizontal Location (m)')
plt.ylabel('Depth (m)')
plt.colorbar(im,fraction=0.034, pad=0.06)
plt.grid(linestyle='--', linewidth=1, alpha=1, color='k')
plt.savefig(os.path.join(save_dir, 'wave-result.eps'), format='eps', bbox_inches='tight', dpi=300)

plt.figure()
im = plt.imshow(np.transpose(data_trainB), vmin=-.5, vmax=.5, cmap="Greys", aspect='1', \
	extent=[0,Xstart,Tend,0], interpolation="lanczos")
plt.xlabel('Horizontal Location (m)')
plt.ylabel('Depth (m)')
plt.colorbar(im,fraction=0.034, pad=0.06)
plt.grid(linestyle='--', linewidth=1, alpha=1, color='k')
plt.savefig(os.path.join(save_dir, 'wave-data_trainB.eps'), format='eps', bbox_inches='tight', dpi=300)

plt.figure()
im = plt.imshow(np.transpose(data_trainA-data_train), vmin=-.5, vmax=.5, cmap="Greys", aspect='1', \
	extent=[0,Xstart,Tend,0], interpolation="lanczos")
plt.xlabel('Horizontal Location (m)')
plt.ylabel('Depth (m)')
plt.colorbar(im,fraction=0.034, pad=0.06)
plt.grid(linestyle='--', linewidth=1, alpha=1, color='k')
plt.savefig(os.path.join(save_dir, 'wave-error.eps'), format='eps', bbox_inches='tight', dpi=300)
