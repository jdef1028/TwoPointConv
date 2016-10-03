import caffe
from caffe_net import generateFilter, freqCount, ConvNetToProto, assignParamsToConvNet#, synFromConvNet
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
model = "../../model/model_09262016"
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


targetResponse = net.blobs['response'].data # objective two point correlation function


# Set bounds for each pixel

bounds = []
for i in range(L1):
	for j in range(L2):
		# set bounds for each pixel
		bounds.append((0,1))

VF = np.sum(img)/float(L1)/float(L2) # calculate the VF 


# initialize a random image

init = np.random.randn(L1, L2)

# define the function to be optimized
def f(x):
	# x is the current image 
	x = x.reshape(*net.blobs['data'].data.shape)
	net.forward(data=x) # forward propagation
	fval = 0

	# clear the current gradient, set to 0 
	net.blobs['response'].diff[...] = np.zeros_like(net.blobs['response'].diff)

	val, grad = mse_loss(net.blobs['response'].data.copy, targetResponse)
	fval += val
	net.blobs['response'].diff[:] += grad
	net.backward()
	f_grad = net.blobs['data'].diff.copy()

	return [f_val, np.array(f_grad.ravel(), dtype=float)]






