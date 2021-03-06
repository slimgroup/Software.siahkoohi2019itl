# The importance of transfer learning in seismic modeling and imaging: Gradient conditioning

Codes for generating _gradient conditioning_ results in Siahkoohi, A., Louboutin, M. and Herrmann, F.J., 2019. The importance of transfer learning in seismic modeling and imaging. Geophysics, 84(6), pp.1-30.  doi: [10.1190/geo2019-0056.1](https://doi.org/10.1190/geo2019-0056.1).

## Dataset

Links have been provided in `RunTraining.sh` script to automatically download the training/testing dataset into the necessary directory. Unfortunately, we are not currently allowed to share the BG COMPASS 3D model. But as mentioned, the dataset necessary to reproduce the results will be downloaded upon running `RunTraining.sh`\.

## Script descriptions

`RunTraining.sh`\: script for running training. It will make `model/` and `data/` directory in `/home/ec2-user/` for storing training/testing data and saved neural net checkpoints and final results, respectively. Next, it will train a neural net for the experiment.

`RunTransferLearning.sh`\: script for running transfer learning. It will load the pre-trained neural net and perform transfer learning.

`RunTesting.sh`\: script for testing the trained neural net. It will perform gradient conditioning for all the low-fidelity gradients for both pre-trained and transfer-trained neural net.

`src/main.py`\: constructs `gradient_conditioning` class using given arguments in `RunTraining.sh`\, defined in `model.py` and calls `train` function in the defined  `gradient_conditioning` class.

`src/model.py`: includes `gradient_conditioning` class definition, which involves `train` and `test` functions.

`utilities/showRTM.py`\: Shows the result of mapping by combining mapped gradients and plotting the RTM.


### Running the code


Data can be downloaded automatically by running `RunTraining.sh`\. To perform training, run:


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


To generate and save figures shown in paper for gradient conditioning, run:

```bash

bash utilities/genRTM.sh

```


The saving directory can be changed by modifying `savePath` variable in `utilities/genRTM.sh`\.


## Questions

Please contact alisk@gatech.edu for further questions.


## Author

Ali Siahkoohi
