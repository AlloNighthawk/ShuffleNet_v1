# ShuffleNet_v1
This repository is an Pytorch implementation of ShuffleNet_v1.
The original articles: [ShuffleNet_v1](https://arxiv.org/pdf/1707.01083.pdf)

## Environments
* Python 3.5.2
* PyTorch 1.0.1

## Dataset:
* Cifar-10

## HOW TO RUN THE CODE
just run `shufflenet_cifar10.py`

## Attention:
* The images of Cifar-10 will be padded since the size of ShuffleNet_v1's inputs is (224,224,3) while the size of images of Cifar-10 is (32, 32, 3).
