#!/home/xiaolin/anaconda2/bin/python

import numpy as np
from math import sqrt
import os
import subprocess
import cPickle as pickle
class weightsCal():
    def __init__(self, l1):
    	# l1 is the max distance between the two points studied

        self.l1 = l1
        self.weightsHash = {}

    def findWeights(self):
        # find all possible weights for CNN
        l1 = self.l1
        for x1 in range(l1):
        	for y1 in range(l1): # point 1
        	    for x2 in range(l1):
                	for y2 in range(l1):# point 2
                		distance = int(round(sqrt((x1-x2)**2 + (y1-y2)**2)))
                        #print x1,y1,x2,y2
                        if distance <= l1:
                        	# only consider the segments whose length is smaller than the specified range
                            CNN = np.zeros((abs(x1 - x2) + 1, abs(y1 - y2) + 1))
                            CNN[0][0] = 1
                            CNN[-1][-1] = 1
                            if distance not in self.weightsHash:
                            	# new distance appears
                            	self.weightsHash[distance] = [CNN]

                            else:
                            	# check if the filter is already there
                                flag = False
                            	for filters in self.weightsHash[distance]:
                                    # since all the CNN weight matrix only has its upper left and lower right pixel to be 1
                                    # the simpliest way for matrix comparison is to compare their shape
                                    if filters.shape == CNN.shape:
                                    	flag = True # already exist
                                
                                if not flag:
                                	# this filter has not been added to the hash
                                	# add it

                                	self.weightsHash[distance].append(CNN)

    def showNum(self):
    	print "The maximum length of vector is", self.l1
    	print "The number of distances is", len(self.weightsHash)
    	for dis in self.weightsHash:
    		print "Distance: ", dis, " number of filter: ", len(self.weightsHash[dis])
    def dumpWeights(self, root):
        if not os.path.isdir("./"+root+str(self.l1)):
           subprocess.Popen("mkdir ./"+root+str(self.l1), shell=True)
        with open("./"+root+str(self.l1)+"/weightsHash.pickle", "w+") as f:
        	pickle.dump(self.weightsHash, f)

"""
if __name__ == "__main__":
	A = twoPointCorrelationWeightsCal(50)
	A.findWeights()
	A.showNum()
	if A.l1 < 10:
		print A.weightsHash
	A.dumpWeights()"""


   
                            



