import numpy as np
import sys
import math

def computeSj(data, data_mean):
	sjarr = []
	for column in range(data.shape[1]):
		sum = 0
		for e in data[:, column]:
			sum = sum + pow(e - data_mean[column],2)
		sjarr.append(sum)
	return sjarr

def computeSy(labels, labels_mean):
	syval = 0
	for val in labels:
		syval = syval + pow((val-labels_mean),2)
	return syval
	
def computeCarr(data, data_mean, labels, labels_mean):
	carr =[]
	for column in range(data.shape[1]):
		sum = 0
		j = 0
		for e in data[:,column]:
			sum = sum + ((e - data_mean[column])*(labels[j] - labels_mean))
			j = j+1
		carr.append(sum)
	return carr
	
def computeRj(data, carr, sjarr, syval):
	rjarr = []
	for k in range(data.shape[1]):
		rjarr.append((carr[k]/math.sqrt(sjarr[k] * syval), k))
	return rjarr
	

if __name__ == "__main__":
	if len(sys.argv) !=4:
		print("usage :", sys.argv[0], "datafile labelsfile outputfile")
		sys.exit()
	else:
		data = np.genfromtxt(sys.argv[1], delimiter =',', dtype='float')
		labels = np.genfromtxt(sys.argv[2], delimiter =',', dtype='float')
		
		labels_mean = labels.mean(axis = 0)
		data_mean = np.mean(data, axis = 0)
		
		sjarr = computeSj(data, data_mean)
		syval = computeSy(labels, labels_mean)
		carr = computeCarr(data, data_mean, labels, labels_mean)
		rjarr = computeRj(data, carr, sjarr, syval)
		
		rjarr.sort(reverse=True)
		result = []
		result.append(rjarr[0][1])
		result.append(rjarr[1][1])
		
		output = data[:,result]
		
		with open(sys.argv[3], mode = 'w', newline='') as op:
			np.savetxt(op, output , delimiter=",",fmt='%.2f')