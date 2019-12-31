import numpy as np
import h5py
import os
from devito.logger import info
from devito import TimeFunction, clear_cache
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import Model, RickerSource, Receiver, TimeAxis
from math import floor
from scipy.interpolate import griddata
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path', dest='data_path', type=str, default='/home/ec2-user/data', help='raw data path')
parser.add_argument('--save_dir', dest='save_dir', type=str, default='/home/ec2-user/data', help='saving directory')
args = parser.parse_args()
data_path  = args.data_path
save_dir = args.save_dir

origin = (0., 0.)
spacing=(7.5, 7.5)
tn=1100.
nbpml=40
# Define your vp in km/sec (x, z)

vp = np.fromfile(os.path.join(data_path, 'vp_marmousi_bi'),
            dtype='float32', sep="")
vp = np.reshape(vp, (1601, 401))
# vp = vp[400:1401, 0:401]
shape=[401, 301]

values = np.zeros([vp.shape[0]*vp.shape[1], ])
points = np.zeros([vp.shape[0]*vp.shape[1], 2])

k = 0
for indx in range(0, vp.shape[0]):
    for indy in range(0, vp.shape[1]):
        values[k] = vp[indx, indy]
        points[k, 0] = indx
        points[k, 1] = indy

        k = k + 1


# nx, ny = shape[0], shape[1]
X, Y = np.meshgrid(np.array(np.linspace(730, 1017, shape[0])), np.array(np.linspace(60, 172, shape[1])))

int_vp = griddata(points, values, (X, Y), method='cubic')
int_vp = np.transpose(int_vp)
vp = int_vp

# create model
model = Model(origin, spacing, shape, 2, vp, nbpml=nbpml)
# Derive timestepping from model spacing
dt = model.critical_dt
t0 = 0.0
nt = int(1 + (tn-t0) / dt)  # Number of timesteps
time = np.linspace(t0, tn, nt)  # Discretized time axis


datasize0 = int(np.shape(range(0, shape[0], 4))[0])
datasize1 = int(np.shape(range(100, nt, 20))[0])
datasize = datasize0*datasize1


strTrainA = os.path.join(save_dir, 'Wavefield_Marmousi_pml_401x301_730-1017_60-172_4k_20kp100_A_train.hdf5')
strTrainB = os.path.join(save_dir, 'Wavefield_Marmousi_pml_401x301_730-1017_60-172_4k_20kp100_B_train.hdf5')
dataset_train = "train_dataset"

file_trainA = h5py.File(strTrainA, 'w-')
datasetA = file_trainA.create_dataset(dataset_train, (datasize, shape[0]+2*nbpml, shape[1]+2*nbpml))

file_trainB = h5py.File(strTrainB, 'w-')
datasetB = file_trainB.create_dataset(dataset_train, (datasize, shape[0]+2*nbpml, shape[1]+2*nbpml))

num_rec = 601
rec_samp = np.linspace(0., model.domain_size[0], num=num_rec);
rec_samp = rec_samp[1]-rec_samp[0]


time_range = TimeAxis(start=t0, stop=tn, step=dt)
src = RickerSource(name='src', grid=model.grid, f0=0.025, time_range=time_range, space_order=1, npoint=1)
src.coordinates.data[0, :] = np.array([1*spacing[0], 2*spacing[1]]).astype(np.float32)

rec = Receiver(name='rec', grid=model.grid, time_range=time_range, npoint=num_rec)
rec.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0], num=num_rec)
rec.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]

solverbad = AcousticWaveSolver(model, source=src, receiver=rec, kernel='OT2', isic=True,
        space_order=2, freesurface=False)
solvergood = AcousticWaveSolver(model, source=src, receiver=rec, kernel='OT2', isic=True,
        space_order=20, freesurface=False)

ulocgood = TimeFunction(name="u", grid=model.grid, time_order=2, space_order=20, save=nt)
ulocbad = TimeFunction(name="u", grid=model.grid, time_order=2, space_order=2, save=nt)


kk = 0
for xsrc in range(0, shape[0], 4):

    clear_cache()

    ulocgood.data.fill(0.)
    ulocbad.data.fill(0.)

    src.coordinates.data[0, :] = np.array([xsrc*spacing[0], 2*spacing[1]]).astype(np.float32)
    rec.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0], num=num_rec)
    rec.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]

    _, ulocgood, _ = solvergood.forward(m=model.m, src=src, time=nt-1, save=True)
    _, ulocbad, _  = solverbad.forward(m=model.m, src=src, time=nt-1, save=True)

    datasetA[kk:(kk+datasize1), :, :] = np.array(ulocgood.data[range(100, nt, 20), :, :])
    datasetB[kk:(kk+datasize1), :, :] = np.array(ulocbad.data[range(100, nt, 20), :, :])

    kk = kk + datasize1



file_trainA.close()
file_trainB.close()
