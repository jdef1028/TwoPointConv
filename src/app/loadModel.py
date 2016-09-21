import caffe
from caffe_net import generateFilter, freqCount, ConvNetToProto, assignParamsToConvNet

model = "../../model/l_1"
model += ".prototxt"

print "== Now calculate weights filters =="
weightHash = generateFilter(1)

print "== Now composte ConvNet =="
ConvNetToProto(weightHash, 1, [1,1,5,5], model)

print "== Now compute frequency counts =="
freqHash = freqCount(5, 5)

print "== load net =="

net = caffe.Net(model, caffe.TEST)

print "== Associate the weights filter to net =="

net = assignParamsToConvNet(net, weightHash, freqHash)


