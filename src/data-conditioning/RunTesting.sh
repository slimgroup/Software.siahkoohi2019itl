# #!/bin/bash -l

experiment_name=DataCorrection_transfer_learning
experiment_name_pretraining=DataCorrection_pretraining
repo_name=importance-of-transfer-learning

path_script=$HOME/$repo_name/src/data-conditioning/src
path_data=$HOME/data
path_model=$HOME/model/$experiment_name
path_model_pretraining=$HOME/model/$experiment_name_pretraining

CUDA_VISIBLE_DEVICES=0 python $path_script/main.py --experiment_dir=data_cond --phase test --batch_size 1 \
	--checkpoint_dir $path_model/checkpoint --sample_dir $path_model/sample --log_dir $path_model/log \

CUDA_VISIBLE_DEVICES=0 python $path_script/main.py --experiment_dir=data_cond --phase test --batch_size 1 \
	--checkpoint_dir $path_model_pretraining/checkpoint --sample_dir $path_model_pretraining/sample \
	--log_dir $path_model_pretraining/log