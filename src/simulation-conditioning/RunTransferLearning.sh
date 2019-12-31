# #!/bin/bash -l

experiment_name=SimulationCorrection_transfer_learning
experiment_name_pretraining=SimulationCorrection_pretraining
repo_name=importance-of-transfer-learning

path_script=$HOME/$repo_name/src/transfer-learning/simulation-conditioning/src
path_data=$HOME/data
path_model=$HOME/model/$experiment_name

cp -r $HOME/model/$experiment_name_pretraining $path_model

CUDA_VISIBLE_DEVICES=0 python $path_script/main.py --experiment_dir=simulation_cond --phase train --batch_size 1 \
	--epoch 11 --epoch_step 5  --save_freq 100  --print_freq 10 \
	--checkpoint_dir $path_model/checkpoint --sample_dir $path_model/sample --log_dir $path_model/log \
	--transfer 1