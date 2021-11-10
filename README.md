# zkCNN

## Introduction

This is a GKR-based zero-knowledge proof for CNN reference, containing some widely used network such as LeNet5, vgg11 and vgg16.



## Requirement

- C++14
- cmake >= 3.10
- GMP library



## Input Format

The input has two part which are data and weight in the matrix.

### Data Part

There are two cases supported in this repo.

- **Single picture**

  Then the picture is a vector reshaped from its original matrix by
  
  ![formula1](https://render.githubusercontent.com/render/math?math=ch_{in}%20%5Ccdot%20h\times%20w)
  
  where ![formula2](https://render.githubusercontent.com/render/math?math=ch_{in}) is the number of channel, ![formula3](https://render.githubusercontent.com/render/math?math=h) is the height, ![formula4](https://render.githubusercontent.com/render/math?math=w) is the width.

  

- **Multiply picture**

  This solve the case when the user wants to infer multiple pictures by the same network. Then the picture is a vector reshaped from its original matrix by
  
  ![formula5](https://render.githubusercontent.com/render/math?math=n_{pic}%20\times%20ch_{in}%20\times%20h%20\times%20w)
  
  where ![formula6](https://render.githubusercontent.com/render/math?math=n_{pic}) is the number of pictures, ![formula7](https://render.githubusercontent.com/render/math?math=ch_{in}) is the number of channel, ![formula8](https://render.githubusercontent.com/render/math?math=h) is the height, ![formula9](https://render.githubusercontent.com/render/math?math=w) is the width.

### Weight Part

This part is for weight in the neural network, which contains

- convolution kernel of size ![formula10](https://render.githubusercontent.com/render/math?math=ch_{out}%20\times%20ch_{in}%20\times%20m%20\times%20m)

  where ![formula11](https://render.githubusercontent.com/render/math?math=ch_{out}) and ![formula12](https://render.githubusercontent.com/render/math?math=ch_{in}) are the number of output and input channels, ![formula13](https://render.githubusercontent.com/render/math?math=m) is the sideness of the kernel (here we only support square kernel).

- convolution bias of size ![formula16](https://render.githubusercontent.com/render/math?math=ch_{out})

- fully-connected kernel of size ![formula14](https://render.githubusercontent.com/render/math?math=ch_{in}\times%20ch_{out})


- fully-connected bias of size ![formula15](https://render.githubusercontent.com/render/math?math=ch_{out})


All the input above are scanned one by one.

## Experiment Script
### Clone the repo
To run the code, make sure you clone with
``` bash
git clone --recurse-submodules git@github.com:TAMUCrypto/zkCNN.git
```
since the polynomial commitment is included as a submodule.

### Run a demo of vgg11
The script to run vgg11 model (please run the script in ``script/`` directory).
``` bash
./demo.sh
```

- The input data is in ``data/vgg11/``.
- The experiment evaluation is ``output/single/demo-result.txt``.
- The inference result is ``output/single/vgg11.cifar.relu-1-infer.csv``.

## Polynomial Commitment

Here we implement a hyrax polynomial commitment based on BLS12-381 elliptic curve. It is a submodule and someone who is interested can refer to this repo [hyrax-bls12-381](https://github.com/TAMUCrypto/hyrax-bls12-381).

