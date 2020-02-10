# #!/bin/bash -l

experiment_name=DataCorrection_transfer_learning
experiment_name_pretraining=DataCorrection_pretraining
repo_name=importance-of-transfer-learning

path_script=$HOME/$repo_name/src/data-conditioning/src
path_data=$HOME/data
path_model=$HOME/model/$experiment_name
path_model_pretraining=$HOME/model/$experiment_name_pretraining

savePath=$path_model/test
savePath_pretraining=$path_model_pretraining/test

set -e

python showSRME.py --hdf5path $path_model/sample  --save_dir $savePath
python showSRME.py --hdf5path $path_model_pretraining/sample  --save_dir $savePath_pretraining