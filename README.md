# zkCNN

## Introduction

This is the implementation of [this paper](https://eprint.iacr.org/2021/673), which is a GKR-based zero-knowledge proof for CNN reference, containing some common CNN models such as LeNet5, vgg11 and vgg16.



## Requirement

- C++14
- cmake >= 3.10
- GMP library



## Input Format

The input has two part which are data and weight in the matrix.

### Data Part

This part is the picture data, a vector reshaped from its original matrix by

![formula1](https://render.githubusercontent.com/render/math?math=ch_{in}%20%5Ccdot%20h\times%20w)

where ![formula2](https://render.githubusercontent.com/render/math?math=ch_{in}) is the number of channel, ![formula3](https://render.githubusercontent.com/render/math?math=h) is the height, ![formula4](https://render.githubusercontent.com/render/math?math=w) is the width.

### Weight Part

This part is the set of parameters in the neural network, which contains

- convolution kernel of size ![formula10](https://render.githubusercontent.com/render/math?math=ch_{out}%20\times%20ch_{in}%20\times%20m%20\times%20m)

  where ![formula11](https://render.githubusercontent.com/render/math?math=ch_{out}) and ![formula12](https://render.githubusercontent.com/render/math?math=ch_{in}) are the number of output and input channels, ![formula13](https://render.githubusercontent.com/render/math?math=m) is the sideness of the kernel (here we only support square kernel).

- convolution bias of size ![formula16](https://render.githubusercontent.com/render/math?math=ch_{out}).

- fully-connected kernel of size ![formula14](https://render.githubusercontent.com/render/math?math=ch_{out}\times%20ch_{in}).

- fully-connected bias of size ![formula15](https://render.githubusercontent.com/render/math?math=ch_{out}).

## Experiment Script
### Clone the repo
To run the code, make sure you clone with
``` bash
git clone --recurse-submodules git@github.com:TAMUCrypto/zkCNN.git
```
since the polynomial commitment is included as a submodule.

### Run a demo of LeNet5
The script to run LeNet5 model (please run the script in ``script/`` directory).
``` bash
./demo_lenet.sh
```

- The input data is in ``data/lenet5.mnist.relu.max/``.
- The experiment evaluation is ``output/single/demo-result-lenet5.txt``.
- The inference result is ``output/single/lenet5.mnist.relu.max-1-infer.csv``.


### Run a demo of vgg11
The script to run vgg11 model (please run the script in ``script/`` directory).
``` bash
./demo_vgg.sh
```

- The input data is in ``data/vgg11/``.
- The experiment evaluation is ``output/single/demo-result.txt``.
- The inference result is ``output/single/vgg11.cifar.relu-1-infer.csv``.

## Polynomial Commitment

Here we implement a [hyrax polynomial commitment](https://eprint.iacr.org/2017/1132.pdf) based on BLS12-381 elliptic curve. It is a submodule and someone who is interested can refer to this repo [hyrax-bls12-381](https://github.com/TAMUCrypto/hyrax-bls12-381).

## Reference
- [zkCNN: Zero knowledge proofs for convolutional neural network predictions and accuracy](https://doi.org/10.1145/3460120.3485379).
  Liu, T., Xie, X., & Zhang, Y. (CCS 2021).

- [Doubly-efficient zksnarks without trusted setup](https://doi.org/10.1109/SP.2018.00060). Wahby, R. S., Tzialla, I., Shelat, A., Thaler, J., & Walfish, M. (S&P 2018)

- [Hyrax](https://github.com/hyraxZK/hyraxZK.git)

- [mcl](https://github.com/herumi/mcl)