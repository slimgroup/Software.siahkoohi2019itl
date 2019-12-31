import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import argparse
np.random.seed(seed=19)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--hdf5path', dest='hdf5path', type=str, default='/home/ec2-user/model', help='path')
parser.add_argument('--save_dir', dest='save_dir', type=str, default='default', help='saving directory')

args = parser.parse_args()

hdf5path  = args.hdf5path
hdf5name = os.path.join(hdf5path, 'mapping_result.hdf5')
save_dir = args.save_dir

def model_topMute(image, mute_end, length):
    mute_start = mute_end - length
    damp = np.zeros([image.shape[0]])
    damp[0:mute_start-1] = 0.
    damp[mute_end+1:] = 1.
    taper_length = mute_end - mute_start + 1
    taper = (1. + np.sin((np.pi*np.array(range(0,taper_length-1)))/(taper_length - 1)-np.pi/2.))/2.
    damp[mute_start:mute_end] = taper
    for j in range(0, image.shape[1]):
        image[:,j] = image[:,j]*damp
    return image

def depth_scaling(image):
    shape = image.shape

    kernel = np.zeros(image.shape)
    scale = np.sqrt(np.linspace(0., 5., image.shape[0]))

    for i in range(shape[1]):
        kernel[:, i] = scale

    image = image * kernel
    return image



strName = hdf5name
dataset_name = "result"

fileName = h5py.File(strName, 'r')
data_num = fileName[dataset_name].shape[0]
data_numA = fileName[dataset_name + 'A'].shape[0]
data_numB = fileName[dataset_name + 'B'].shape[0]

data_train = fileName[dataset_name][:,:,:]
data_trainA = fileName[dataset_name + 'A'][:,:,:]
data_trainB = fileName[dataset_name + 'B'][:,:,:]


# data_train = np.delete(data_train, np.arange(0, 201, 20), 0)

image = np.sum(data_train, axis=0)/np.linalg.norm(np.sum(fileName[dataset_name][:,:,:], axis=0).reshape(-1), np.inf)
imageA = np.sum(data_trainA, axis=0)/np.linalg.norm(np.sum(fileName[dataset_name + 'A'][:,:,:], axis=0).reshape(-1), np.inf)
imageB = np.sum(data_trainB, axis=0)/np.linalg.norm(np.sum(fileName[dataset_name + 'B'][:,:,:], axis=0).reshape(-1), np.inf)

spacing = (12.5, 6.)
shape = imageA.shape
Xlength = shape[0] * spacing[0]
Zlength =  shape[1] * spacing[1]

font = {'family' : 'sans-serif',
        'size'   : 8}
import matplotlib
matplotlib.rc('font', **font)


image = model_topMute((np.transpose(image)), 39, 5)
imageA = model_topMute((np.transpose(imageA)), 39, 5)
imageB = model_topMute((np.transpose(imageB)), 34, 5)


image = depth_scaling(image)
imageA = depth_scaling(imageA)
imageB = depth_scaling(imageB)


if save_dir == 'default':
    save_dir = os.path.join(hdf5path, 'figs')
    if os.path.isdir(save_dir)==False:
        os.mkdir(save_dir)
        print(save_dir)


xticks = np.arange(0, Xlength, 500)
yticks = np.arange(2000, 0, -500)


fig, ax = plt.subplots()
im = ax.imshow(imageA, vmin=-7e-2, vmax=7e-2, aspect='1', cmap="Greys", extent=[0,Xlength,Zlength,0])
#plt.title('High-fidelity RTM')
ax.set_xlabel('Horizontal Location (m)')
ax.set_ylabel('Depth (m)')
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_yticks([1500, 1300, 742], minor=True)
ax.set_xticks(np.arange(0, 5500, 500), minor=False)
ax.yaxis.grid(True, which='minor', linestyle='--', linewidth=1, alpha=1, color='k')
plt.colorbar(im,fraction=0.0197, pad=0.015)
plt.savefig(os.path.join(save_dir, 'figure-3a.eps'), format='eps', bbox_inches='tight', dpi=300)

fig, ax = plt.subplots()
im = ax.imshow(imageB, vmin=-2.5e-1, vmax=2.5e-1, aspect='1', cmap="Greys", extent=[0,Xlength,Zlength,0])
#plt.title('Low-fidelity RTM')
ax.set_xlabel('Horizontal Location (m)')
ax.set_ylabel('Depth (m)')
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_yticks([1500, 1300, 742], minor=True)
ax.set_xticks(np.arange(0, 5500, 500), minor=False)
ax.yaxis.grid(True, which='minor', linestyle='--', linewidth=1, alpha=1, color='k')
plt.colorbar(im,fraction=0.0197, pad=0.015)
plt.savefig(os.path.join(save_dir, 'figure-3b.eps'), format='eps', bbox_inches='tight', dpi=300)

fig, ax = plt.subplots()
im = ax.imshow(image, vmin=-7e-2, vmax=7e-2, aspect='1', cmap="Greys", extent=[0,Xlength,Zlength,0])
#plt.title('Corrected RTM w/ transfer training shot locations')
ax.set_xlabel('Horizontal Location (m)')
ax.set_ylabel('Depth (m)')
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_yticks([1500, 1300, 742], minor=True)
ax.set_xticks(np.arange(0, 5500, 500), minor=False)
ax.yaxis.grid(True, which='minor', linestyle='--', linewidth=1, alpha=1, color='k')
plt.colorbar(im,fraction=0.0197, pad=0.015)
plt.savefig(os.path.join(save_dir, 'figure-3c.eps'), format='eps', bbox_inches='tight', dpi=300)


