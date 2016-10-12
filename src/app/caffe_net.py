#!/home/xiaolin/anaconda2/bin/python

import caffe
import copy
from caffe import layers as L
from caffe import params as P
from math import sqrt
import numpy as np
import subprocess
from scipy.ndimage.filters import gaussian_filter


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

def freqCount(L1, L2):
	# calculate the total number of occurrence for each length in the image
	# input:
	#    L1: the width of the image
	#    L2: the length of the image
	freqHash = {}
	for x1 in range(L1):
		for y1 in range(L2):
			# point 1
			for x2 in range(L1):
				for y2 in range(L2):
                # point 2
					distance = int(round(sqrt((x1-x2)**2 + (y1-y2)**2)))
					if distance not in freqHash:
							# first occurrence
						if distance != 0:
							freqHash[distance] = 1./2
						else:
							freqHash[distance] = 1.
					else:
							# the distance has occurred in the previous calculation
						if distance != 0:
							freqHash[distance] += 1./2
						else:
							freqHash[distance] += 1.


	return freqHash




def ConvNetToProto(weightsHash, ll, data_dim, model_path):
	# from the given weightsHash table, construct the CNN for computing two-point correlation function
	# input:
	#   weightsHash: the hash table calculated before which contains all the weights filter on different
	#                length scale
	#   ll: max length of the two-point distance to be considered

	# ========================================================================
	#                        Description                         
	# In this prototxt generation script, the data layer is propagated through
	#     a. Convolutional Filter whose weights are precalculated (with bias = -1 to cancel the partial identification)
	#     b. in-place ReLU layer to remove 0 and -1 in the activation layer
	#     c. fully connected layer with weights [1]*length to sum the responses up
	#     d. concatenating layer to merge results from different filtering
	#     e. fully connected layer to sum things up
	# ========================================================================

    net = caffe.NetSpec()
    net.data = L.DummyData(shape=dict(dim=data_dim))

    num_filter = 0

    sum_layer_repo = []
    for LL in range(ll+1):
		# the Convolutional network on different length scale needs to be constructure separately
        print "Now constructing filters at length " + str(LL) + "."
        print "The number of filters at this length is: " + str(len(weightsHash[LL]))
        num_filter += len(weightsHash[LL]) # accumulate the total number of filters
        print "# of Filters so far: " + str(num_filter)
        
        fc_layer_repo = [] # a temperory storage for all the fc layer names 


        for idx in xrange(1, len(weightsHash[LL])+1):
        	weights = weightsHash[LL][idx - 1]
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
                 ",pad = 0" + \
                 ",weight_filler = dict(type='constant', value=0)" + \
                 ",bias_filler = dict(type='constant', value=-1))"
        	#print conv_command

        	exec(conv_command)

            # construct in-place relu layer

        	relu_command = "net." + relu_layer_name + "= L.ReLU(net." + conv_layer_name +", in_place=True)"

        	#print relu_command

        	exec(relu_command)

            # construct in-place fc layer to sum up
        	fc_command = "net." + fc_layer_name + "=L.InnerProduct(net." + relu_layer_name + ", num_output=1," + \
                 "weight_filler = dict(type='constant', value=1)," + \
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
        sum_layer_repo.append("net."+sum_layer_name)

        sum_command = "net." + sum_layer_name +"=L.InnerProduct(net.concat_" + str(LL) + ", num_output=1," + \
        			  "weight_filler = dict(type='constant', value=1), " + \
        			  "bias_filler = dict(type='constant', value=0))"
        #print sum_command
        exec(sum_command)

    overall_concat_layer_name = "net_response_components"
    overall_concat_command = overall_concat_layer_name + "=["
    for response_ele in sum_layer_repo[:-1]:
    	overall_concat_command += response_ele + ", "
    overall_concat_command += sum_layer_repo[-1]
    overall_concat_command += "]"
    exec(overall_concat_command)

    overall_concat_command2 = "net.response = L.Concat(*" + overall_concat_layer_name + ")"
    exec(overall_concat_command2)


    #print net
    with open(model_path, "w") as f:
        f.write(str(net.to_proto()))

    # reformat the model file to enable backward diff on data
    
    subprocess.Popen("sed -i -e '1,13d' "+model_path, shell=True).wait()
    subprocess.Popen("sed -i '1i name: \"Two point correlation ConvNet\"' " + model_path, shell=True).wait()
    subprocess.Popen("sed -i '2i input: \"data\"' " + model_path, shell=True).wait()
    subprocess.Popen("sed -i '3i input_dim: " + str(data_dim[0]) +"' " + model_path, shell=True).wait()
    subprocess.Popen("sed -i '4i input_dim: " + str(data_dim[1]) +"' " + model_path, shell=True).wait()
    subprocess.Popen("sed -i '5i input_dim: " + str(data_dim[2]) +"' " + model_path, shell=True).wait()
    subprocess.Popen("sed -i '6i input_dim: " + str(data_dim[3]) +"' " + model_path, shell=True).wait()
    subprocess.Popen("sed -i '7i force_backward: true' " + model_path, shell=True).wait()

    # with the force_backward statement activated, the derivative w.r.t data mode has been on.
    



def assignParamsToConvNet(net, weightsHash, freqHash):
	# after loading the composed structure, assign the parameters to the params

	ll = max(weightsHash.keys()) # maximum distance to be considered

	for LL in range(ll+1):
		numOfFilter = len(weightsHash[LL]) # the number of filters at distance LL

		for idx in xrange(1, len(weightsHash[LL])+1):
			# iterate over every stored weights

			# Note: here we only need to update the weights matrix in the convolutional layer
			#       all other layers have been handled appropriately in the net construction step

			weight = copy.deepcopy(weightsHash[LL][idx - 1]) # extract the weight filter to be assigned

			conv_layer_name = "conv_" + str(LL) + "_" + str(idx) # convolutional filter name composed before

            # make sure the dimension is matching between the params and the weights matrix

			d11, d12 = weight.shape

			d21, d22 = net.params[conv_layer_name][0].data.shape[2:]

			assert (d11 == d21) # first dimension examination
			assert (d12 == d22) # second dimension examination
			assert type(weight) == np.ndarray

            # before the assignments of the params in the conv layer, we need to convert the 2D weights filter
            # to 4D blob
		
			weight_blob = weight[np.newaxis, np.newaxis, :, :]

            # assign the weigts to the params matrix
			net.params[conv_layer_name][0].data[...] = weight_blob
			if LL == 0:
				net.params[conv_layer_name][1].data[...] = 0
            # in addition to the assignment of the convolutional filter, we alse need to use the freqHash to rescale
            # the filterred result to frequency (probability) of the two-point correlation
		sum_layer_name = "sum_" + str(LL)

		freqNormalizer = float(freqHash[LL])

		net.params[sum_layer_name][0].data[...] /= freqNormalizer

	return net


def mse_loss(activation, target, verbose=0):
	# mse loss on two point correlation function
	

	loss = 1./2 * (activation - target) ** 2 # mse
	loss = loss.sum(1)
	grad = (activation - target)
	if verbose == 1:
		print "Activation: ", activation
		print "Target: ", target
		print "Loss: ", loss
	return [loss, grad]



def deviateImg(img, option='Gaussian'):
	# after reaching the local minimum, apply this method to deviate the solution a little bit 
	# for escaping from local minumum
	# input:
	#    img: 2D - numpy array that represents the optimized image
	#    option: method of deviation

	# assertions
	assert type(img) == np.ndarray
	
	options = ['Gaussian']
	assert option in options
	img = img.astype(float)

	# apply the chosen approach to deviation
	if option == 'Gaussian':
		# Gaussian blur
		# parameters:
		sigma = 0.2

		ret = gaussian_filter(img, sigma)

		return ret 







       







