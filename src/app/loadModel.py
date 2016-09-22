import caffe
from caffe_net import generateFilter, freqCount, ConvNetToProto, assignParamsToConvNet
import numpy as np
from scipy.io import loadmat

# === load binary image from the specified path ===
mat_path = "../../data/img1.mat"
img_var = "img_out"

data = loadmat(mat_path)
img = data[img_var]

"""img = np.array([[1,0,0,0,1],
	            [1,1,1,0,0],
	            [0,0,1,1,0],
	            [1,1,1,0,0],
	            [0,0,0,0,0]])"""

L1, L2 = img.shape
ll = min(L1, L2) / 2
# ==== compose the caffe net and associate with appropriate weights ===
model = "../../model/model_09222016"
model += ".prototxt"

print "== Now calculate weights filters =="
weightHash = generateFilter(ll)

print "== Now composte ConvNet =="
ConvNetToProto(weightHash, ll, [1,1,L1,L2], model)

print "== Now compute frequency counts =="
freqHash = freqCount(L1, L2)
print freqHash
print "== load net =="

net = caffe.Net(model, caffe.TEST)

print "== Associate the weights filter to net =="

net = assignParamsToConvNet(net, weightHash, freqHash)



# ===== Import image ===== 


img_blob = img[np.newaxis, np.newaxis, :, :]

net.blobs['data'].data[...] = img_blob
net.forward()

print net.blobs['response'].data

