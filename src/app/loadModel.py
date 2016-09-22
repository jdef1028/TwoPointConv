import caffe
from caffe_net import generateFilter, freqCount, ConvNetToProto, assignParamsToConvNet
import numpy as np
from scipy.io import loadmat

# === load binary image from the specified path ===
mat_path = "../../data/img1.mat" # file path of the mat file which contains the binary image
img_var = "img_out" # variable name of the image in the .mat file

data = loadmat(mat_path)
img = data[img_var]

"""img = np.array([[1,0,0,0,1],
	            [1,1,1,0,0],
	            [0,0,1,1,0],
	            [1,1,1,0,0],
	            [0,0,0,0,0]])"""

L1, L2 = img.shape
ll = 4
# ==== compose the caffe net and associate with appropriate weights ===
model = "../../model/model_09222016"
model += ".prototxt"

# calculate the weights filters
# TODO: Intermediate file dump and load to save computational time

weightHash = generateFilter(ll)


# compose the ConvNet structure in caffe
# TODO: prototxt file examination (possibly write and load operation) to save computational time
ConvNetToProto(weightHash, ll, [1,1,L1,L2], model)

# compute the occurrence weights terms for penalty on the output
freqHash = freqCount(L1, L2)
print freqHash

# load the ConvNet structure from .prototxt
net = caffe.Net(model, caffe.TEST)


# assign appropriate weights to the convolutional nets
net = assignParamsToConvNet(net, weightHash, freqHash)



# ===== Import image ===== 


img_blob = img[np.newaxis, np.newaxis, :, :]

net.blobs['data'].data[...] = img_blob
net.forward()

targetResponse = net.blobs['response'].data

