import numpy as np
import h5py
import os
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path', dest='data_path', type=str, default='/home/ec2-user/data', help='raw data path')
parser.add_argument('--save_dir', dest='save_dir', type=str, default='/home/ec2-user/data', help='saving directory')
args = parser.parse_args()
data_path  = args.data_path
save_dir = args.save_dir

##1
strTrainA1 = os.path.join(data_path, 'Wavefield_Marmousi_pml_401x301_0-287_200-312_20k_30kp100_A_train.hdf5')
strTrainB1 = os.path.join(data_path, 'Wavefield_Marmousi_pml_401x301_0-287_200-312_20k_30kp100_B_train.hdf5')
dataset_train = "train_dataset"

file_trainA1 = h5py.File(strTrainA1, 'r')
file_trainB1 = h5py.File(strTrainB1, 'r')

data_numA1 = file_trainA1[dataset_train].shape[0]
data_numB1 = file_trainB1[dataset_train].shape[0]

data_trainA1 = file_trainA1[dataset_train]
data_trainB1 = file_trainB1[dataset_train]

shape0 = file_trainA1[dataset_train].shape[1]
shape1 = file_trainA1[dataset_train].shape[2]




#############################################################
##2

strTrainA2 = os.path.join(data_path, 'Wavefield_Marmousi_pml_401x301_1000-1287_120-232_4k_20kp100_A_train.hdf5')
strTrainB2 = os.path.join(data_path, 'Wavefield_Marmousi_pml_401x301_1000-1287_120-232_4k_20kp100_B_train.hdf5')
dataset_train = "train_dataset"

file_trainA2 = h5py.File(strTrainA2, 'r')
file_trainB2 = h5py.File(strTrainB2, 'r')

data_numA2 = file_trainA2[dataset_train].shape[0]
data_numB2 = file_trainB2[dataset_train].shape[0]

data_trainA2 = file_trainA2[dataset_train]
data_trainB2 = file_trainB2[dataset_train]

shape0 = file_trainA2[dataset_train].shape[1]
shape1 = file_trainA2[dataset_train].shape[2]

####################################################################
##3

strTrainA3 = os.path.join(data_path, 'Wavefield_Marmousi_pml_401x301_1013-1300_38-150_6k_30kp100_A_train.hdf5')
strTrainB3 = os.path.join(data_path, 'Wavefield_Marmousi_pml_401x301_1013-1300_38-150_6k_30kp100_B_train.hdf5')
dataset_train = "train_dataset"

file_trainA3 = h5py.File(strTrainA3, 'r')
file_trainB3 = h5py.File(strTrainB3, 'r')

data_numA3 = file_trainA3[dataset_train].shape[0]
data_numB3 = file_trainB3[dataset_train].shape[0]

data_trainA3 = file_trainA3[dataset_train]
data_trainB3 = file_trainB3[dataset_train]

shape0 = file_trainA3[dataset_train].shape[1]
shape1 = file_trainA3[dataset_train].shape[2]

###################################################################
##4

strTrainA4 = os.path.join(data_path, 'Wavefield_Marmousi_pml_401x301_500-787_130-242_4k_20kp100_A_train.hdf5')
strTrainB4 = os.path.join(data_path, 'Wavefield_Marmousi_pml_401x301_500-787_130-242_4k_20kp100_B_train.hdf5')
dataset_train = "train_dataset"

file_trainA4 = h5py.File(strTrainA4, 'r')
file_trainB4 = h5py.File(strTrainB4, 'r')

data_numA4 = file_trainA4[dataset_train].shape[0]
data_numB4 = file_trainB4[dataset_train].shape[0]

data_trainA4 = file_trainA4[dataset_train]
data_trainB4 = file_trainB4[dataset_train]

shape0 = file_trainA4[dataset_train].shape[1]
shape1 = file_trainA4[dataset_train].shape[2]

###################################################################
##5

strTrainA5 = os.path.join(data_path, 'Wavefield_Marmousi_pml_401x301_730-1017_60-172_4k_20kp100_A_train.hdf5')
strTrainB5 = os.path.join(data_path, 'Wavefield_Marmousi_pml_401x301_730-1017_60-172_4k_20kp100_B_train.hdf5')
dataset_train = "train_dataset"

file_trainA5 = h5py.File(strTrainA5, 'r')
file_trainB5 = h5py.File(strTrainB5, 'r')

data_numA5 = file_trainA5[dataset_train].shape[0]
data_numB5 = file_trainB5[dataset_train].shape[0]

data_trainA5 = file_trainA5[dataset_train]
data_trainB5 = file_trainB5[dataset_train]

shape0 = file_trainA5[dataset_train].shape[1]
shape1 = file_trainA5[dataset_train].shape[2]


###################################################################

strTrainACombined = os.path.join(save_dir, 'Wavefield_Marmousi_pml_401x301_combined_A_train.hdf5')
strTrainBCOmbined = os.path.join(save_dir, 'Wavefield_Marmousi_pml_401x301_combined_B_train.hdf5')
dataset_train = "train_dataset"

file_trainACombined = h5py.File(strTrainACombined, 'w-')
datasetACombined = file_trainACombined.create_dataset(dataset_train, \
    (data_numA1+data_numA2+data_numA3+data_numA4+data_numA5, shape0, shape1))

file_trainBCombined = h5py.File(strTrainBCOmbined, 'w-')
datasetBCombined = file_trainBCombined.create_dataset(dataset_train, \
    (data_numB1+data_numB2+data_numB3+data_numB4+data_numB5, shape0, shape1))

kk = 0


datasetACombined[kk:(kk+data_numA1), :, :] = data_trainA1[:,:,:]
datasetBCombined[kk:(kk+data_numA1), :, :] = data_trainB1[:,:,:]

kk = kk + data_numA1


datasetACombined[kk:(kk+data_numA2), :, :] = data_trainA2[:,:,:]
datasetBCombined[kk:(kk+data_numA2), :, :] = data_trainB2[:,:,:]

kk = kk + data_numA2

datasetACombined[kk:(kk+data_numA3), :, :] = data_trainA3[:,:,:]
datasetBCombined[kk:(kk+data_numA3), :, :] = data_trainB3[:,:,:]

kk = kk + data_numA3

datasetACombined[kk:(kk+data_numA4), :, :] = data_trainA4[:,:,:]
datasetBCombined[kk:(kk+data_numA4), :, :] = data_trainB4[:,:,:]

kk = kk + data_numA4

datasetACombined[kk:(kk+data_numA5), :, :] = data_trainA5[:,:,:]
datasetBCombined[kk:(kk+data_numA5), :, :] = data_trainB5[:,:,:]

kk = kk + data_numA5


################################


file_trainACombined.close()
file_trainBCombined.close()
