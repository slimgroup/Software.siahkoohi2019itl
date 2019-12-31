# #!/bin/bash -l

experiment_name=DataCorrection_transfer_learning
experiment_name_pretraining=DataCorrection_pretraining
repo_name=importance-of-transfer-learning

path_script=$HOME/$repo_name/src/transfer-learning/data-conditioning/src
path_data=$HOME/data
path_model=$HOME/model/$experiment_name

cp -r $HOME/model/$experiment_name_pretraining $path_model

CUDA_VISIBLE_DEVICES=0 python $path_script/main.py --experiment_dir=data_cond --phase train --batch_size 1 \
	--epoch 100 --epoch_step 50  --save_freq 100  --print_freq 10 \
	--checkpoint_dir $path_model/checkpoint --sample_dir $path_model/sample --log_dir $path_model/log \
	--transfer 1