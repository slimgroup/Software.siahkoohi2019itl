# #!/bin/bash -l

experiment_name=DataCorrection_pretraining
repo_name=importance-of-transfer-learning

path_script=$HOME/$repo_name/src/data-conditioning/src
path_data=$HOME/data
path_model=$HOME/model/$experiment_name

mkdir $HOME/data
mkdir $HOME/model/
mkdir $path_model

yes | cp -r $path_script/. $path_model

if [ ! -f $path_data/MultipleElimination_Salt_A_1k_WaterDepth10_train_SourceDepth_Freq50.hdf5 ]; then
	wget https://www.dropbox.com/s/e3na3lrmr5ni34h/MultipleElimination_Salt_A_1k_WaterDepth10_train_SourceDepth_Freq50.hdf5 \
		-O $path_data/MultipleElimination_Salt_A_1k_WaterDepth10_train_SourceDepth_Freq50.hdf5
fi

if [ ! -f $path_data/MultipleElimination_Salt_B_1k_WaterDepth10_train_SourceDepth_Freq50.hdf5 ]; then
	wget https://www.dropbox.com/s/nlh4995qa2sydis/MultipleElimination_Salt_B_1k_WaterDepth10_train_SourceDepth_Freq50.hdf5 \
		-O $path_data/MultipleElimination_Salt_B_1k_WaterDepth10_train_SourceDepth_Freq50.hdf5
fi

if [ ! -f $path_data/MultipleElimination_Salt_A_1k_test_SourceDepth_Freq50.hdf5 ]; then
	wget https://www.dropbox.com/s/f515ym2bxlyuwkl/MultipleElimination_Salt_A_1k_test_SourceDepth_Freq50.hdf5 \
		-O $path_data/MultipleElimination_Salt_A_1k_test_SourceDepth_Freq50.hdf5
fi

if [ ! -f $path_data/MultipleElimination_Salt_B_1k_test_SourceDepth_Freq50.hdf5 ]; then
	wget https://www.dropbox.com/s/537dge8h3m68xmt/MultipleElimination_Salt_B_1k_test_SourceDepth_Freq50.hdf5 \
		-O $path_data/MultipleElimination_Salt_B_1k_test_SourceDepth_Freq50.hdf5
fi

CUDA_VISIBLE_DEVICES=0 python $path_script/main.py --experiment_dir=data_cond --phase train --batch_size 1 \
	--checkpoint_dir $path_model/checkpoint --sample_dir $path_model/sample --log_dir $path_model/log \
	--transfer 0

