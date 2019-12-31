# The importance of transfer learning in seismic modeling and imaging: Simulation conditioning

Codes for generating _simulation conditioning_ results in Siahkoohi, A., Louboutin, M. and Herrmann, F.J., 2019. The importance of transfer learning in seismic modeling and imaging. Geophysics, 84(6), pp.1-30.  doi: [10.1190/geo2019-0056.1](https://doi.org/10.1190/geo2019-0056.1).

## Dataset

Links have been provided in `RunTraining.sh` script to automatically download the training/testing dataset into the necessary directory.

Otherwise, scripts and necessary commands to generate data can be found below. For that matter, please get the Marmousi velocity model by running the command below (this can be done automatically by running `RunDataGeneration.sh`\. Details to follow). The Marmousi model we use is obtained from [Devito Codes project](https://github.com/devitocodes)\.

```bash
wget https://github.com/devitocodes/data/raw/master/Simple2D/vp_marmousi_bi
```

## Script descriptions

`RunTraining.sh`\: script for running training. It will make `model/` and `data/` directory in `/home/ec2-user/` for storing training/testing data and saved neural net checkpoints and final results, respectively. Next, it will train a neural net for the experiment.

`RunTransferLearning.sh`\: script for running transfer learning. It will load the pre-trained neural net and perform transfer learning.

`RunTesting.sh`\: script for testing the trained neural net. It will perform simulation conditioning for all the low-fidelity wavefield snapshots for both pre-trained and transfer-trained neural net.

`src/main.py`\: constructs `simulation_conditioning` class using given arguments in `RunTraining.sh`\, defined in `model.py` and calls `train` function in the defined  `simulation_conditioning` class.

`src/model.py`: includes `simulation_conditioning` class definition, which involves `train` and `test` functions.

`utilities/RunDataGeneration.sh`\: Generates training/testing pairs by running scripts in `utilities/data-generation-scripts/` directory and passing arguments to them. Then it combines the datasets by running `utilities/combineHDF5.py`\. It will automatically download the the cropped Marmousi velocity model if it does not exists already.

`utilities/showResults.py`\: Shows the result of mapping by combining mapped gradients and plotting the RTM.


### Running the code

To generate training data using the the cropped Marmousi velocity model, run the command below. You can skip this part if you do not want to generate the data yourself. Data can be downloaded automatically by running `RunTraining.sh`\.

```bash
bash utilities/RunDataGeneration.sh
```


To perform training, run:

```bash
# Running on GPU

bash RunTraining.sh

```


To perform transfer learning, after pre-training,, run:

```bash
# Running on GPU

bash RunTransferLearning.sh

```


To evaluated the pre-trained and transfer-trained neural net on test dataset run the following. It will automatically load the latest checkpoint saved for both neural nets.

```bash
# Running on GPU

bash RunTesting.sh

```


To generate and save figures shown in paper for simulation conditioning, run:

```bash

bash utilities/showResults.sh

```

The saving directory can be changed by modifying `savePath` variable in `utilities/genSRME.sh`\.


## Questions

Please contact alisk@gatech.edu for further questions.


## Author

Ali Siahkoohi
