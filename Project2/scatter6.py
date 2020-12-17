import numpy as np
import sys

if __name__ == "__main__":
	if len(sys.argv) !=4:
		print("usage :", sys.argv[0], "datafile labelsfile outputfile")
		sys.exit()
	else:
		X = np.genfromtxt(sys.argv[1], delimiter =',', dtype='float')
		y = np.genfromtxt(sys.argv[2], delimiter =',', dtype='float')
		
		mean_vectors = []
		for cl in range(1,4):
		    mean_vectors.append(np.mean(X[y==cl], axis=0))
		    print('Mean Vector class %s: %s\n' %(cl, mean_vectors[cl-1]))
			
		overall_mean = np.mean(X, axis=0)

		S_B = np.zeros((4,4))
		for i,mean_vec in enumerate(mean_vectors):  
		    n = X[y==i+1,:].shape[0]
		    mean_vec = mean_vec.reshape(4,1) # make column vector
		    overall_mean = overall_mean.reshape(4,1) # make column vector
		    S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

		print('between-class Scatter Matrix:\n', S_B)
		
		
		eigvals, eigvecs = np.linalg.eigh(S_B@S_B.T)
		eigvecs = eigvecs[:,np.argsort(-eigvals)]
		eigvecs = eigvecs[:,:2]
		result = []
		
		for val in X:
			res = eigvecs.T@val
			result.append(res)
			
		output = np.asarray(result)
		
		with open(sys.argv[3], mode = 'w', newline='') as op:
			np.savetxt(op, output , newline = " ")
			
		