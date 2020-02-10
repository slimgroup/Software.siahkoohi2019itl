# The importance of transfer learning in seismic modeling and imaging

This repository contains source codes for experiemtns in Siahkoohi, A., Louboutin, M. and Herrmann, F.J., 2019. The importance of transfer learning in seismic modeling and imaging. Geophysics, 84(6), pp.1-30.  doi: [10.1190/geo2019-0056.1](https://doi.org/10.1190/geo2019-0056.1).


## Prerequisites

This code has been tested on Deep Learning AMI (Amazon Linux) Version 24.2 (predefined `tensorflow_p36` conda environment) on Amazon Web Services (AWS). We used `g3s.xlarge` instance. Also, we use GCC compiler version 7.3.0.

Links to required datasets (shot records, gradients, and wavefield snapshots) to reproduce the results in the paper are provided in the scripts described below. Unless you want to generate them yourself, feel free to skip installing Devito. If you are interested in generating the training/testing data yourself, the code that generates training/testing data in this experiment is based on a customized [Devito-3.2.0](https://github.com/opesci/devito/releases/tag/v3.2.0) which can be found on my [GitHub repository](https://github.com/alisiahkoohi/devito/tree/customized). Follow the steps below to install the required softwares:

```bash
cd $HOME
git clone https://github.com/alisiahkoohi/devito.git
git clone https://github.com/alisiahkoohi/importance-of-transfer-learning.git

cd devito
git checkout customized
conda env create -f environment.yml
source activate devito-customized
pip install -e .
export DEVITO_ARCH=gnu
export OMP_NUM_THREADS=4
export DEVITO_OPENMP=1

cd $HOME/importance-of-transfer-learning
pip install --user -r  requirements.txt
```

## Repository map

    .
    |
    ├── src                  			# source codes
    │   ├── data-conditioning
    │   ├── gradient-conditioning
    │   ├── simulation-conditioning
    └── ...

* `data-conditioning/`\: source code for generating data conditioning results 

* `gradient-conditioning/`\: source code for generating gradient conditioning results 

* `simulation-conditioning/`\: source code for generating simulation conditioning results 


For more details regarding how to run the experiments, refer to the README.md file in each experiment's directory.

## Citation

If you find this siftware useful in your research, please cite:

```bibtex
@article{siahkoohi2019transfer,
    author={Siahkoohi, Ali and Louboutin, Mathias and Herrmann, Felix J.},
    title={The importance of transfer learning in seismic modeling and imaging},
    month={7},
    year={2019},
    doi = {10.1190/geo2019-0056.1},
    journal = {{Geophysics}},
    number = {6},
    pages = {A47--A52},
    publisher = {SEG},
    volume = {84}
}
```


## Questions

Please contact alisk@gatech.edu for further questions.

## Acknowledgments

The authors thank Xiaowei Hu for his open-access [repository](https://github.com/xhujoy/CycleGAN-tensorflow) on GitHub. Our software implementation built on this work.

## Author

Ali Siahkoohi
