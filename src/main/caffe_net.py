import caffe
import copy
from caffe import layers as L
from caffe import params as P
def generateFilter(ll):
	# generate weight filters within the given range (ll)

	# input:
	# 	ll: max length of two-point distance to be considered
	# output:
	# 	weightsHash: all possible weightfilters

	weightsHash = {}

	for x1 in range(ll):
		for y1 in range(ll):
			for x2 in range(ll):
				for y2 in range(ll):
					# iteratation for the two points (x1, y1) and (x2, y2)

					distance = int(round(sqrt((x1-x2)**2 + (y1-y2)**2)))

					if distance <= ll:
						# The length of the segment is qualified. Go ahead to record
						canvas = np.zeros(abs(x1-x2), abs(y1-y2))
						# Here we consider two possibilities
						# 1. the two points are located at the upper left and lower right of the mask (diagnal)
						CNN1 = copy.deepcopy(canvas)
						CNN1[0][0] = 1
						CNN1[-1][-1] = 1
						# 2. the two points are located at the lower left and upper right of the mask (deputy diagnal)
						CNN2 = copy.deepcopy(canvas)
						CNN2[0][-1] = 1
						CNN2[-1][0] = 1

					if distance not in weightsHash:
						# new distance key
						weightsHash[distance] = [CNN1, CNN2]
					else:
						# the distance key is already added, check if the filter is already there
						# CNN1 and CNN2 should be either both in the hash or neither
						flag = False
						for existing_filter in weightsHash[distance]:
							if existing_filter.shape == CNN1.shape:
								flag = True
						if not flag:
							weightsHash[distance].append(CNN1)
							weightsHash[distance].append(CNN2)

	return weightsHash

def ConvNetProto(weightsHash, ll, data):
	# from the given weightsHash table, construct the CNN for computing two-point correlation function
	# input:
	#   weightsHash: the hash table calculated before which contains all the weights filter on different
	#                length scale
	#   ll: max length of the two-point distance to be considered
    net = caffe.NetSpec()
    net.data = data
    num_filter = 0
    for L in range(ll):
		# the Convolutional network on different length scale needs to be constructure separately
        print "Now constructing filters at length " + str(ll) + "."
        print "The number of filters at this length is: " + str(len(weightsHash[L]))
        num_filter += len(weightsHash[L]) # accumulate the total number of filters
        print "# of Filters so far: " + str(num_filter)
        
        fc_layer_repo = [] # a temperory storage for all the fc layer names 


        for idx in range(len(weightsHash[L])):
        	weights = weightsHash[L][idx]
        	kernel_height, kernel_width = weights.shape
        	
        	conv_layer_name = "conv_" + str(L) + "_" + str(idx)
        	relu_layer_name = "relu_" + str(L) + "_" + str(idx)
        	fc_layer_name = "fc_" + str(L) + "_" + str(idx)
        	fc_layer_repo.append(fc_layer_name)

            # construct convolutional layer
        	exec("net." + conv_layer_name + "= L.Convolution(net.data" +
            	 ",kernel_h =" + str(kernel_height) +
            	 ",kernel_w =" + str(kernel_width) +
                 ",num_output = 1" +
                 ",stride = 1" +
                 ",pad = 1" +
                 ",weight_filler = dict(type='constant', value=0)")

            # construct in-place relu layer
        	exec("net." + relu_layer_name + "= L.ReLU(net." + conv_layer_name +", in_place=True)")

            # construct in-place fc layer to sum up
        	exec("net." + fc_layer_name + "=L.InnerProduct(net." + fc_layer_name+", num_output=1," +
                 "weight_filler = dict(type='constant', value=0)," +
                 "bias_filler = dict(type='constant', valule=0))")

        sum_layer_name = "sum_layer_" + str(L)

        num_fc_layer = len(fc_layer_repo)

        exec("net.concat_" + str(L) +" = L.Concat(*fc_layer_repo)")

    with open("tpConvModel.prototxt", "w") as f:
    	f.write(str(net.to_protp()))









       







