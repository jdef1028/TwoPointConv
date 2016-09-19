#!/home/xiaolin/anaconda2/bin/python

import caffe
import copy
from caffe import layers as L
from caffe import params as P
from math import sqrt
import numpy as np

def generateFilter(ll):
	# generate weight filters within the given range (ll)

	# input:
	# 	ll: max length of two-point distance to be considered
	# output:
	# 	weightsHash: all possible weightfilters

	weightsHash = {}

	for x1 in range(ll+1):
		for y1 in range(ll+1):
			for x2 in range(ll+1):
				for y2 in range(ll+1):
					# iteratation for the two points (x1, y1) and (x2, y2)

					distance = int(round(sqrt((x1-x2)**2 + (y1-y2)**2)))

					if distance <= ll:
						# The length of the segment is qualified. Go ahead to record
						canvas = np.zeros((abs(x1-x2)+1, abs(y1-y2)+1), dtype=int)
						# Here we consider two possibilities
						# 1. the two points are located at the upper left and lower right of the mask (diagnal)
						CNN1 = copy.deepcopy(canvas)
						CNN1[0][0] = 1
						CNN1[-1][-1] = 1
						# 2. the two points are located at the lower left and upper right of the mask (deputy diagnal)
						CNN2 = copy.deepcopy(canvas)
						CNN2[0][-1] = 1
						CNN2[-1][0] = 1

					if distance == 0:
						weightsHash[distance] = [CNN1]

					elif distance not in weightsHash:
						# new distance key
						if not np.array_equal(CNN1, CNN2):
							weightsHash[distance] = [CNN1, CNN2]
						else:
							weightsHash[distance] = [CNN1]
					else:
						# the distance key is already added, check if the filter is already there
						# CNN1 and CNN2 should be either both in the hash or neither
						flag1 = False
						flag2 = False
						for existing_filter in weightsHash[distance]:
							if np.array_equal(existing_filter, CNN1):
								flag1 = True
							if np.array_equal(existing_filter, CNN2):
								flag2 = True

						if not flag1:
							# add the filter weights to convolutional filter repository at the specific distance
							weightsHash[distance].append(CNN1)
						if not np.array_equal(CNN1, CNN2):
							if not flag2:
								weightsHash[distance].append(CNN2)


	return weightsHash

def ConvNetToProto(weightsHash, ll, data_dim):
	# from the given weightsHash table, construct the CNN for computing two-point correlation function
	# input:
	#   weightsHash: the hash table calculated before which contains all the weights filter on different
	#                length scale
	#   ll: max length of the two-point distance to be considered
    net = caffe.NetSpec()
    net.data = L.DummyData(shape=dict(dim=data_dim))
    
    num_filter = 0
    for LL in range(ll):
		# the Convolutional network on different length scale needs to be constructure separately
        print "Now constructing filters at length " + str(ll) + "."
        print "The number of filters at this length is: " + str(len(weightsHash[LL]))
        num_filter += len(weightsHash[LL]) # accumulate the total number of filters
        print "# of Filters so far: " + str(num_filter)
        
        fc_layer_repo = [] # a temperory storage for all the fc layer names 


        for idx in range(len(weightsHash[LL])):
        	weights = weightsHash[LL][idx]
        	kernel_height, kernel_width = weights.shape
        	
        	conv_layer_name = "conv_" + str(LL) + "_" + str(idx)
        	#print "ConvLayer: ", conv_layer_name

        	relu_layer_name = "relu_" + str(LL) + "_" + str(idx)
        	#print "ReLULayer: ", relu_layer_name

        	fc_layer_name = "fc_" + str(LL) + "_" + str(idx)
        	#print "fcLayer:", fc_layer_name
        	fc_layer_repo.append("net."+fc_layer_name)

            # construct convolutional layer
        	conv_command = "net." + conv_layer_name + "= L.Convolution(net.data" + \
            	 ",kernel_h =" + str(kernel_height) + \
            	 ",kernel_w =" + str(kernel_width) + \
                 ",num_output = 1" + \
                 ",stride = 1" + \
                 ",pad = 1" + \
                 ",weight_filler = dict(type='constant', value=0))"
        	#print conv_command

        	exec(conv_command)

            # construct in-place relu layer

        	relu_command = "net." + relu_layer_name + "= L.ReLU(net." + conv_layer_name +", in_place=True)"

        	#print relu_command

        	exec(relu_command)

            # construct in-place fc layer to sum up
        	fc_command = "net." + fc_layer_name + "=L.InnerProduct(net." + relu_layer_name + ", num_output=1," + \
                 "weight_filler = dict(type='constant', value=0)," + \
                 "bias_filler = dict(type='constant', value=0))"
        	#print fc_command
        	exec(fc_command)

        concat_layer_name = "concat_" + str(LL) + "_Layers"
        concat_command1 = concat_layer_name + "=["
        for concat_ele in fc_layer_repo[:-1]:
        	concat_command1 += concat_ele + ", "
        concat_command1 += fc_layer_repo[-1]
        concat_command1 += "]"
        #print concat_command1
        exec(concat_command1)

        concat_command2 = "net.concat_" + str(LL) + "=L.Concat(*" + concat_layer_name +")"
        #print concat_command2
        exec(concat_command2)

        sum_layer_name = "sum_" + str(LL)

        sum_command = "net." + sum_layer_name +"=L.InnerProduct(net.concat_" + str(LL) + ", num_output=1," + \
        			  "weight_filler = dict(type='constant', value=0), " + \
        			  "bias_filler = dict(type='constant', value=0))"
        #print sum_command
        exec(sum_command)
    #print net
    with open("tpConvModel.prototxt", "w") as f:
    	f.write(str(net.to_proto()))









       







