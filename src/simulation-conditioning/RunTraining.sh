# #!/bin/bash -l

experiment_name=SimulationCorrection_pretraining
repo_name=importance-of-transfer-learning

path_script=$HOME/$repo_name/src/transfer-learning/simulation-conditioning/src
path_data=$HOME/data
path_model=$HOME/model/$experiment_name

mkdir $HOME/data
mkdir $HOME/model/
mkdir $path_model

yes | cp -r $path_script/. $path_model

if [ ! -f $path_data/Wavefield_Marmousi_401x301_combined_A_train.hdf5 ]; then
	wget https://www.dropbox.com/s/m9a5tu12zebqi1j/Wavefield_Marmousi_401x301_combined_A_train.hdf5 \
		-O $path_data/Wavefield_Marmousi_401x301_combined_A_train.hdf5
fi

if [ ! -f $path_data/Wavefield_Marmousi_401x301_combined_B_train.hdf5 ]; then
	wget https://www.dropbox.com/s/vp1a14l728dzm00/Wavefield_Marmousi_401x301_combined_B_train.hdf5 \
		-O $path_data/Wavefield_Marmousi_401x301_combined_B_train.hdf5
fi

if [ ! -f $path_data/Wavefield_Marmousi_401x301_500-1300_0-312_2k_40kp200_A_test.hdf5 ]; then
	wget https://www.dropbox.com/s/gqycuokzzyauzzj/Wavefield_Marmousi_401x301_500-1300_0-312_2k_40kp200_A_test.hdf5 \
		-O $path_data/Wavefield_Marmousi_401x301_500-1300_0-312_2k_40kp200_A_test.hdf5
fi

if [ ! -f $path_data/Wavefield_Marmousi_401x301_500-1300_0-312_2k_40kp200_B_test.hdf5 ]; then
	wget https://www.dropbox.com/s/fbmjgfxi6d3goof/Wavefield_Marmousi_401x301_500-1300_0-312_2k_40kp200_B_test.hdf5 \
		-O $path_data/Wavefield_Marmousi_401x301_500-1300_0-312_2k_40kp200_B_test.hdf5
fi

if [ ! -f $path_data/Wavefield_Marmousi_401x301_500-1300_0-312_20kp1_20kp220_A_train.hdf5 ]; then
	wget https://www.dropbox.com/s/k4esk7ywk6ryb9t/Wavefield_Marmousi_401x301_500-1300_0-312_20kp1_20kp220_A_train.hdf5 \
		-O $path_data/Wavefield_Marmousi_401x301_500-1300_0-312_20kp1_20kp220_A_train.hdf5
fi

if [ ! -f $path_data/Wavefield_Marmousi_401x301_500-1300_0-312_20kp1_20kp220_B_train.hdf5 ]; then
	wget https://www.dropbox.com/s/8u1giuwj18fnr4p/Wavefield_Marmousi_401x301_500-1300_0-312_20kp1_20kp220_B_train.hdf5 \
		-O $path_data/Wavefield_Marmousi_401x301_500-1300_0-312_20kp1_20kp220_B_train.hdf5
fi

CUDA_VISIBLE_DEVICES=0 python $path_script/main.py --experiment_dir=simulation_cond --phase train --batch_size 1 \
	--checkpoint_dir $path_model/checkpoint --sample_dir $path_model/sample --log_dir $path_model/log \
	--transfer 0
