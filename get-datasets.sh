#!/bin/bash

mkdir datasets
cd datasets
wget http://deeplearning.net/data/mnist/mnist.pkl.gz
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar xzvf cifar-10-python.tar.gz
cd ..
echo "Downloaded Datasets!"