#!/home/xiaolin/anaconda2/bin/python

import unittest
from app.caffe_net import *
import numpy as np
from collections import Counter
class TestCaffeNetUtil(unittest.TestCase):

	def test_weights_filter_calculation(self):
		ret = generateFilter(1)
		self.assertTrue(np.array_equal(ret[0], [np.array([[1]])]))


		expected_ret1 = [np.array([[1,1]]),
			             np.array([[1],[1]]),
			             np.array([[1,0],
			                       [0,1]]),
			             np.array([[0,1],
			                       [1,0]])]
		self.assertTrue(listCompare(ret[1], expected_ret1))
			                                      
def listCompare(list1, list2):
	for ele1 in list1:
		ele1_flag = False
		for ele2 in list2:
			if np.array_equal(ele1, ele2):
				ele1_flag = True
		if not ele1_flag:
			return False
	return True
		                



if __name__ == "__main__":
	unittest.main()