# #!/bin/bash -l

repo_name=importance-of-transfer-learning

path_script=$HOME/$repo_name/src/simulation-conditioning/utilities
path_data=$HOME/data

mkdir $HOME/data

if [ ! -f $path_data/vp_marmousi_bi ]; then
	wget https://github.com/devitocodes/data/raw/master/Simple2D/vp_marmousi_bi \
		-O $path_data/vp_marmousi_bi
fi

set -e

python $path_script/data-generation-scripts/Wavefield_Marmousi_pml_401x301_0-287_200-312_20k_30kp100_A_train.py \
	--data_path $path_data --save_dir $path_data

python $path_script/data-generation-scripts/Wavefield_Marmousi_pml_401x301_500-787_130-242_4k_20kp100_A_train.py \
	--data_path $path_data --save_dir $path_data

python $path_script/data-generation-scripts/Wavefield_Marmousi_pml_401x301_500-1300_0-312_2k_40kp200_A_test.py \
	--data_path $path_data --save_dir $path_data

python $path_script/data-generation-scripts/Wavefield_Marmousi_pml_401x301_500-1300_0-312_20kp1_20kp220_A_train.py \
	--data_path $path_data --save_dir $path_data

python $path_script/data-generation-scripts/Wavefield_Marmousi_pml_401x301_730-1017_60-172_4k_20kp100_A_train.py \
	--data_path $path_data --save_dir $path_data

python $path_script/data-generation-scripts/Wavefield_Marmousi_pml_401x301_1000-1287_120-232_4k_20kp100_A_train.py \
	--data_path $path_data --save_dir $path_data

python $path_script/data-generation-scripts/Wavefield_Marmousi_pml_401x301_1013-1300_38-150_6k_30kp100_A_train.py \
	--data_path $path_data --save_dir $path_data


python $path_script/combineHDF5.py --data_path $path_data --save_dir $path_data