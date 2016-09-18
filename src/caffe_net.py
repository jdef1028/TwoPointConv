import caffe
import copy
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



