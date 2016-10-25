# ConvAE-DeSTIN
A DeSTIN implementation using stacked convolutional auto-encoders

## Install Dependencies

See http://deeplearning.net/software/theano/install_ubuntu.html

## Install Cifar Dataset

Execute this from root project folder:

    wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

(MD5: c58f30108f718f92721af3b95e74349a)

## Running experiments

On my machine (using only CPUs) it can be launched by the following command:

    cd scripts
    THEANO_FLAGS='floatX=float32'  python convae_destin_1.py 
