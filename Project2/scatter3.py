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
			
		S_W = np.zeros((4,4))
		for cl,mv in zip(range(1,4), mean_vectors):
		    class_sc_mat = np.zeros((4,4))                  
		    for row in X[y == cl]:
		        row, mv = row.reshape(4,1), mv.reshape(4,1) 
		        class_sc_mat += (row-mv).dot((row-mv).T)
		    S_W += class_sc_mat                             
			
		print('within-class Scatter Matrix:\n', S_W)
		
		eigvals, eigvecs = np.linalg.eigh(S_W@S_W.T)
		eigvecs = eigvecs[:,np.argsort(eigvals)]
		eigvecs = eigvecs[:,:2]
		result = []
		
		for val in X:
			res = eigvecs.T@val
			result.append(res)
			
		output = np.asarray(result)
		
		with open(sys.argv[3], mode = 'w', newline='') as op:
			np.savetxt(op, output , newline = " ")
			
		