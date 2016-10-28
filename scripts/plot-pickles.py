#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""

!!! Run this in the result folder you want to look at.

"""

import sys
sys.path.append("../../")

import numpy as np
import matplotlib
matplotlib.use('AGG') 
import matplotlib.pyplot as plt
import cPickle as pickle
import time
import os
import theano
import theano.tensor as T
import gzip

import scae_destin.datasets as ds
from scae_destin.fflayers import ReLULayer
from scae_destin.fflayers import SoftmaxLayer
from scae_destin.convnet import ReLUConvLayer
from scae_destin.convnet import SigmoidConvLayer
from scae_destin.model import ConvAutoEncoder
from scae_destin.convnet import MaxPooling
from scae_destin.convnet import Flattener
from scae_destin.model import FeedForward
from scae_destin.optimize import gd_updates
from scae_destin.cost import mean_square_cost
from scae_destin.cost import categorical_cross_entropy_cost
from scae_destin.cost import L2_regularization

for epoch in xrange(9):
	## Load layers for this epoch...
	layers = pickle.load(open('unsupervised_layers_epoch_%d.pkl' % int(epoch+1), 'rb'))

	for layer in layers:
		##### Output filter images code#1
		data = np.zeros(((layer.filter_size[0] + 3) * layer.num_filters,(layer.filter_size[1] + 3) * layer.num_channels), dtype=np.float32 )
		data[::] = -0.1 #some activations are below 0, to keep borders black set them to -0.1
	
		for filter_index in xrange(layer.num_filters):
		    for channel_index in xrange(layer.num_channels):
		        arr = layer.filters.eval()[filter_index][channel_index]
		        for p in xrange(layer.filter_size[0]):
		            for q in xrange(layer.filter_size[1]):
		                data[filter_index*(3+layer.filter_size[1])+p][channel_index*(3+layer.filter_size[1])+q] = arr[p][q]
		            
		#print data.shape
		plt.imshow(data, cmap = plt.get_cmap('gray'), interpolation='nearest')
		plt.axis('off')
		plt.savefig("unsupervised_epoch_%d_layer_%d.png" % (int(epoch+1), layers.index(layer)) , dpi=300)
	    



