#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("..")

import numpy as np
import matplotlib
matplotlib.use('PS') 
import matplotlib.pyplot as plt
import time
import theano
import theano.tensor as T

import utilities.datasets as ds
import utilities.results as res

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

##initialize folder for results
results = res.ResultSaver();

##load data
[(train_set_x, train_set_y), 
 (valid_set_x, valid_set_y), 
 (test_set_x, test_set_y)] = ds.load_mnist('../datasets/mnist.pkl.gz');

##### For running fast locally
train_set_x = train_set_x[:10]
train_set_y = train_set_y[:10]
###############################

print "> The data is loaded in shared memory"

start_time=time.time()

n_epochs=1
batch_size=10
nkerns=100

n_train_batches=train_set_x.eval().shape[0]/batch_size
n_test_batches=test_set_x.eval().shape[0]/batch_size

X=T.matrix("data")
y=T.ivector("label")
idx=T.lscalar()
corruption_level=T.fscalar()

print "> The data is loaded!"

images=X.reshape((batch_size, 1, 28, 28))

layer_0_en=ReLUConvLayer(filter_size=(7,7),
                         num_filters=50,
                         num_channels=1,
                         fm_size=(28,28),
                         batch_size=batch_size)
                                                  
layer_0_de=SigmoidConvLayer(filter_size=(7,7),
                            num_filters=1,
                            num_channels=50,
                            fm_size=(22,22),
                            batch_size=batch_size,
                            border_mode="full")
                         
layer_1_en=ReLUConvLayer(filter_size=(5,5),
                         num_filters=50,
                         num_channels=50,
                         fm_size=(22,22),
                         batch_size=batch_size)
                                                   
layer_1_de=SigmoidConvLayer(filter_size=(5,5),
                            num_filters=50,
                            num_channels=50,
                            fm_size=(18,18),
                            batch_size=batch_size,
                            border_mode="full")

layer_2_en=ReLUConvLayer(filter_size=(5,5),
                         num_filters=50,
                         num_channels=50,
                         fm_size=(18,18),
                         batch_size=batch_size)
                                                   
layer_2_de=SigmoidConvLayer(filter_size=(5,5),
                            num_filters=50,
                            num_channels=50,
                            fm_size=(14,14),
                            batch_size=batch_size,
                            border_mode="full")

layer_3_en=ReLUConvLayer(filter_size=(3,3),
                         num_filters=50,
                         num_channels=50,
                         fm_size=(14,14),
                         batch_size=batch_size)
                                                   
layer_3_de=SigmoidConvLayer(filter_size=(3,3),
                            num_filters=50,
                            num_channels=50,
                            fm_size=(12,12),
                            batch_size=batch_size,
                            border_mode="full")

model_0=ConvAutoEncoder(layers=[layer_0_en, layer_0_de])
out_0=model_0.fprop(images, corruption_level=corruption_level)
cost_0=mean_square_cost(out_0[-1], images)+L2_regularization(model_0.params, 0.005)
updates_0=gd_updates(cost=cost_0, params=model_0.params, method="sgd", learning_rate=0.1)

model_1=ConvAutoEncoder(layers=[layer_1_en, layer_1_de])
out_1=model_1.fprop(out_0[0], corruption_level=corruption_level)
cost_1=mean_square_cost(out_1[-1], out_0[0])+L2_regularization(model_1.params, 0.005)
updates_1=gd_updates(cost=cost_1, params=model_1.params, method="sgd", learning_rate=0.1)

model_2=ConvAutoEncoder(layers=[layer_2_en, layer_2_de])
out_2=model_2.fprop(out_1[0], corruption_level=corruption_level)
cost_2=mean_square_cost(out_2[-1], out_1[0])+L2_regularization(model_2.params, 0.005)
updates_2=gd_updates(cost=cost_2, params=model_2.params, method="sgd", learning_rate=0.1)

model_3=ConvAutoEncoder(layers=[layer_3_en, layer_3_de])
out_3=model_3.fprop(out_2[0], corruption_level=corruption_level)
cost_3=mean_square_cost(out_3[-1], out_2[0])+L2_regularization(model_3.params, 0.005)
updates_3=gd_updates(cost=cost_3, params=model_3.params, method="sgd", learning_rate=0.1)

train_0=theano.function(inputs=[idx, corruption_level],
                        outputs=[cost_0],
                        updates=updates_0,
                        givens={X: train_set_x[idx * batch_size: (idx + 1) * batch_size]})

train_1=theano.function(inputs=[idx, corruption_level],
                        outputs=[cost_1],
                        updates=updates_1,
                        givens={X: train_set_x[idx * batch_size: (idx + 1) * batch_size]})

train_2=theano.function(inputs=[idx, corruption_level],
                        outputs=[cost_2],
                        updates=updates_2,
                        givens={X: train_set_x[idx * batch_size: (idx + 1) * batch_size]})

train_3=theano.function(inputs=[idx, corruption_level],
                        outputs=[cost_3],
                        updates=updates_3,
                        givens={X: train_set_x[idx * batch_size: (idx + 1) * batch_size]})
                      
print "> The 4-layer model is built"

corr={}
corr[0]=corr[1]=corr[2]=corr[3]=np.random.uniform(low=0.1, high=0.2, size=1).astype("float32")

min_cost={0:None,
          1:None,
          2:None,
          3:None}

corr_best={0:corr[0],
           1:corr[0],
           2:corr[0],
           3:corr[0]}

max_iter={0:0,
          1:0,
          2:0,
          3:0}

print "> Beginning training..."

epoch = 0
while (epoch < n_epochs):
    epoch = epoch + 1
    c_0 = c_1 = c_2 = c_3 = []
    for batch_index in xrange(n_train_batches):
      for rep in xrange(8):
        train_cost=train_3(batch_index, corr_best[3][0])
        c_3.append(train_cost)
        train_cost=train_2(batch_index, corr_best[2][0])
        c_2.append(train_cost)
        train_cost=train_1(batch_index, corr_best[1][0])
        c_1.append(train_cost)
        train_cost=train_0(batch_index, corr_best[0][0])
        c_0.append(train_cost)
        
    if min_cost[0]==None:
        min_cost[0]=np.mean(c_0)
    else:
        if (np.mean(c_0)<min_cost[0]*0.5) or (max_iter[0]>=20):
            min_cost[0]=np.mean(c_0)
            corr_best[0][0]=corr[0]
            corr[0]=np.random.uniform(low=corr_best[0][0], high=corr_best[0][0]+0.1, size=1).astype("float32")
            max_iter[0]=0
        else:
            max_iter[0]+=1

    if min_cost[1]==None:
            min_cost[1]=np.mean(c_1)
    else:
        if (np.mean(c_1)<min_cost[1]*0.5) or (max_iter[1]>=20):
            min_cost[1]=np.mean(c_1)
            corr_best[1][0]=corr[1]
            corr[1]=np.random.uniform(low=corr_best[1][0], high=corr_best[1][0]+0.1, size=1).astype("float32")
            max_iter[1]=0
        else:
            max_iter[1]+=1

    if min_cost[2]==None:
            min_cost[2]=np.mean(c_2)
    else:
        if (np.mean(c_2)<min_cost[2]*0.5) or (max_iter[2]>=20):
            min_cost[2]=np.mean(c_2)
            corr_best[2][0]=corr[2]
            corr[2]=np.random.uniform(low=corr_best[2][0], high=corr_best[2][0]+0.1, size=1).astype("float32")
            max_iter[2]=0
        else:
            max_iter[2]+=1

    if min_cost[3]==None:
            min_cost[3]=np.mean(c_3)
    else:
        if (np.mean(c_3)<min_cost[3]*0.5) or (max_iter[3]>=20):
            min_cost[3]=np.mean(c_3)
            corr_best[3][0]=corr[3]
            corr[3]=np.random.uniform(low=corr_best[3][0], high=corr_best[3][0]+0.1, size=1).astype("float32")
            max_iter[3]=0
        else:
            max_iter[3]+=1
            
    print '> Training epoch %d, cost ' % epoch, np.mean(c_0), str(corr_best[0][0]), min_cost[0], max_iter[0]
    print '                          ', np.mean(c_1), str(corr_best[1][0]), min_cost[1], max_iter[1]
    print '                          ', np.mean(c_2), str(corr_best[2][0]), min_cost[2], max_iter[2]
    print '                          ', np.mean(c_3), str(corr_best[3][0]), min_cost[3], max_iter[3]

    results.dump([layer_0_en, layer_1_en, layer_2_en, layer_3_en], "unsupervised_layers_epoch_%d.zip" % epoch)

print "> The model is trained"

################################## BUILD SUPERVISED MODEL #######################################
                     
flattener=Flattener()
layer_5=ReLULayer(in_dim=50*12*12,
                  out_dim=1000)
layer_6=SoftmaxLayer(in_dim=1000,
                     out_dim=10)

model_sup=FeedForward(layers=[layer_0_en, layer_1_en, layer_2_en, layer_3_en, flattener, layer_5, layer_6])
 
out_sup=model_sup.fprop(images)
cost_sup=categorical_cross_entropy_cost(out_sup[-1], y)
updates=gd_updates(cost=cost_sup, params=model_sup.params, method="sgd", learning_rate=0.1)
 
train_sup=theano.function(inputs=[idx],
                          outputs=cost_sup,
                          updates=updates,
                          givens={X: train_set_x[idx * batch_size: (idx + 1) * batch_size],
                                  y: train_set_y[idx * batch_size: (idx + 1) * batch_size]})
 
test_sup=theano.function(inputs=[idx],
                         outputs=model_sup.layers[-1].error(out_sup[-1], y),
                         givens={X: test_set_x[idx * batch_size: (idx + 1) * batch_size],
                                 y: test_set_y[idx * batch_size: (idx + 1) * batch_size]})
                              
print "> The supervised model is being trained"

n_epochs=1
test_record=np.zeros((n_epochs, 1))
epoch = 0
while (epoch < n_epochs):
    epoch+=1
    for minibatch_index in xrange(n_train_batches):
        mlp_minibatch_avg_cost = train_sup(minibatch_index)
         
        iteration = (epoch - 1) * n_train_batches + minibatch_index
         
        if (iteration + 1) % n_train_batches == 0:
            test_losses = [test_sup(i) for i in xrange(n_test_batches)]
            test_record[epoch-1] = np.mean(test_losses)
             
            print(('>>>> Training epoch %i, minibatch %i/%i, test error %f %%') %
                  (epoch, minibatch_index + 1, n_train_batches, test_record[epoch-1] * 100.))

            results.dump([layer_0_en, layer_1_en, layer_2_en, layer_3_en, flattener, layer_5, layer_6], "supervised_layers_epoch_%d.zip" % epoch)

print "> The supervised model is trained!"
 
results.dump(test_record, "test_result.zip");

print ">  All done. Exported results in " + result_dir
