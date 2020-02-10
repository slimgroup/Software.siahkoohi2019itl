# #!/bin/bash -l

experiment_name=GradientCorrection_pretraining
repo_name=importance-of-transfer-learning

path_script=$HOME/$repo_name/src/gradient-conditioning/src
path_data=$HOME/data
path_model=$HOME/model/$experiment_name

mkdir $HOME/data
mkdir $HOME/model/
mkdir $path_model

yes | cp -r $path_script/. $path_model

if [ ! -f $path_data/gradientCorrection_A_freq40-wide_picked_train.hdf5 ]; then
	wget https://www.dropbox.com/s/skeh9b4o8sfpa7k/gradientCorrection_A_freq40-wide_picked_train.hdf5 \
		-O $path_data/gradientCorrection_A_freq40-wide_picked_train.hdf5
fi

if [ ! -f $path_data/gradientCorrection_B_freq40-wide_picked_train.hdf5 ]; then
	wget https://www.dropbox.com/s/7sqxrjqcvi6rg1y/gradientCorrection_B_freq40-wide_picked_train.hdf5 \
		-O $path_data/gradientCorrection_B_freq40-wide_picked_train.hdf5
fi

if [ ! -f $path_data/gradientCorrection_A_freq40-wide_test.hdf5 ]; then
	wget https://www.dropbox.com/s/tqimsk2l8nwur81/gradientCorrection_A_freq40-wide_test.hdf5 \
		-O $path_data/gradientCorrection_A_freq40-wide_test.hdf5
fi

if [ ! -f $path_data/gradientCorrection_B_freq40-wide_test.hdf5 ]; then
	wget https://www.dropbox.com/s/id1jep90asaahw4/gradientCorrection_B_freq40-wide_test.hdf5 \
		-O $path_data/gradientCorrection_B_freq40-wide_test.hdf5
fi

CUDA_VISIBLE_DEVICES=0 python $path_script/main.py --experiment_dir=gradient_cond --phase train --batch_size 1 \
	--checkpoint_dir $path_model/checkpoint --sample_dir $path_model/sample --log_dir $path_model/log \
	--transfer 0

