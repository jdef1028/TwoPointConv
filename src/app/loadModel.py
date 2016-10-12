import caffe
from caffe_net import generateFilter, freqCount, ConvNetToProto, assignParamsToConvNet, mse_loss, deviateImg
import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
import matplotlib.pyplot as plt
# === load binary image from the specified path ===

mat_path = "../../data/img2.mat" # file path of the mat file which contains the binary image
img_var = "img_out" # variable name of the image in the .mat file

data = loadmat(mat_path)
img = data[img_var]

"""img = np.array([[1,0,0,0,1],
	            [1,1,1,0,0],
	            [0,0,1,1,0],
	            [1,1,1,0,0],
	            [0,0,0,0,0]])"""

L1, L2 = img.shape
ll = 25
# ==== compose the caffe net and associate with appropriate weights ===
model = "../../model/model_10052016"
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


targetResponse = net.blobs['response'].data.copy() # objective two point correlation function
print "The target response is, ", targetResponse

# Set bounds for each pixel

bounds = []
for i in range(L1):
	for j in range(L2):
		# set bounds for each pixel
		bounds.append((0,1))

VF = np.sum(img)/float(L1)/float(L2) # calculate the VF 



# initialize a random image

init = np.random.uniform(low=0.5, high=0.9, size=(L1, L2))
#init = np.random.binomial(1, VF, L1*L2).reshape(L1, -1)

# define the function to be optimized
def f(x):
	# x is the current image 
	x = x.reshape(*net.blobs['data'].data.shape)
	net.forward(data=x) # forward propagation
	f_val = 0

	# clear the current gradient, set to 0 
	net.blobs['response'].diff[...] = np.zeros_like(net.blobs['response'].diff)

	val, grad = mse_loss(net.blobs['response'].data.copy(), targetResponse, verbose=1)
	f_val += val
	net.blobs['response'].diff[:] += grad
	net.backward()
	f_grad = net.blobs['data'].diff.copy()

	return [f_val, np.array(f_grad.ravel(), dtype=float)]
maxiter = 1000
m=100
max_loop = 3
err_tol = 0.0001
error = 100
loop_num = 0
while (loop_num < max_loop) or (error<err_tol):
	minimize_option = {'maxiter': maxiter,
						'maxcor': m,
						'ftol': 0,
						'gtol': 0,
						'maxls': 50}

	ret = minimize(f, init,
					method='L-BFGS-B',
					jac=True,
					bounds=bounds,
					options=minimize_option)

	optimized_structure_float = ret.x.copy()
	optimized_structure_float = optimized_structure_float.reshape((L1, L2))
	print "====== Iteration #" + str(loop_num) + "======"
	print optimized_structure_float

	init = deviateImg(optimized_structure_float, option='Gaussian')
	print init
	net.blobs['data'].data[...] = init
	#print net.blobs['data'].data
	net.forward()
	activation_now = net.blobs['response'].data.copy()
	print "activation: ", activation_now 
	error, _ = mse_loss(activation_now, targetResponse, verbose=0)
	print "error: ", error
	loop_num += 1











