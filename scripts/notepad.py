"""ConvAE Notepad (not for execution, just code snippets)"""


################################## Layer init #######################################

layer_0_en=ReLUConvLayer(filter_size=(7,7),
                         num_filters=50,
                         num_channels=1,
                         fm_size=(32,32),
                         batch_size=batch_size)
                                                  
layer_0_de=SigmoidConvLayer(filter_size=(7,7),
                            num_filters=1,
                            num_channels=50,
                            fm_size=(26,26),
                            batch_size=batch_size,
                            border_mode="full")
                         
model_0=ConvAutoEncoder(layers=[layer_0_en, layer_0_de])
out_0=model_0.fprop(images, corruption_level=corruption_level)
cost_0=mean_square_cost(out_0[-1], images)+L2_regularization(model_0.params, 0.005)
updates_0=gd_updates(cost=cost_0, params=model_0.params, method="sgd", learning_rate=0.1)

train_0=theano.function(inputs=[idx, corruption_level],
                        outputs=[cost_0],
                        updates=updates_0,
                        givens={X: train_set_x[idx * batch_size: (idx + 1) * batch_size]})

















################################## Training #######################################

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
            
    print 'Training epoch %d, cost ' % epoch, np.mean(c_0), str(corr_best[0][0]), min_cost[0], max_iter[0]
    print '                        ', np.mean(c_1), str(corr_best[1][0]), min_cost[1], max_iter[1]
    print '                        '  , np.mean(c_2), str(corr_best[2][0]), min_cost[2], max_iter[2]
    print '                        ' , np.mean(c_3), str(corr_best[3][0]), min_cost[3], max_iter[3]
    











##### Output filter images code#1
def draw_filters_for_layer(layer, filename):
  data = np.zeros(((layer.filter_size[0] + 3) * layer.num_filters,(layer.filter_size[1] + 3) * layer.num_channels), dtype=np.float32 )
  data[::] = -0.1 #some activations are below 0, to keep borders black set them to -0.1

  for filter_index in xrange(layer.num_filters):
      for channel_index in xrange(layer.num_channels):
          arr = layer.filters.get_value()[filter_index][channel_index]
          for p in xrange(layer.filter_size[0]):
              for q in xrange(layer.filter_size[1]):
                  data[filter_index*(3+layer.filter_size[1])+p][channel_index*(3+layer.filter_size[1])+q] = arr[p][q]
              
  print data.shape
  img = smp.toimage(data)
  img.save(filename)

##### Output filter images code#2
for i in xrange(128):
  for j in xrange(5):
    if i > (len(filters[j]) -1):
      continue
    image_adr="/home/profloo/Documents/ConvAE-DeSTIN/scripts/plots/layer_%d_filter_%d.eps" % (j,i)
    plt.imshow(filters[j][i, 0, :, :], cmap = plt.get_cmap('gray'), interpolation='nearest')
    plt.axis('off')
    plt.savefig(image_adr , bbox_inches='tight', pad_inches=0)
    if i%10 == 0:
        print 'completed saving filters till : ', i



