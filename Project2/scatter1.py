import numpy as np
import sys

if __name__ == "__main__":
	if len(sys.argv) !=4:
		print("usage :", sys.argv[0], "datafile labelsfile outputfile")
		sys.exit()
	else:
		data = np.genfromtxt(sys.argv[1], delimiter =',', dtype='float')
		labels = np.genfromtxt(sys.argv[2], delimiter =',', dtype='float')
		
		X = (data-data.mean(axis=0))
		ms_mat = (X.T).dot(X)
		
		eigval, eigvec = np.linalg.eigh(ms_mat@ms_mat.T)
		eigvec = eigvec[:,np.argsort(eigval)]
		eigvec = eigvec[:,:2]
		
		result = []
		for val in data:
			res = eigvec.T@val
			result.append(res)
		output = np.asarray(result)
		with open(sys.argv[3], mode = 'w', newline='') as op:
			np.savetxt(op, output , newline = " ")
			
		