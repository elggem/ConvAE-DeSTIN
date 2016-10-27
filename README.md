# ConvAE-DeSTIN
A DeSTIN implementation using stacked convolutional auto-encoders

## Install Dependencies

See http://deeplearning.net/software/theano/install_ubuntu.html

## Install Datasets

Execute get_datasets.sh from project root.

## Running experiments

On my machine (using only CPUs) it can be launched by the following command:

    cd scripts
    THEANO_FLAGS='floatX=float32'  python convae_destin_1.py 
