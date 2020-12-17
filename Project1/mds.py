import numpy as np
import sys
import math
import pandas as pd

def build_distance_matrix(X, alpha):
    dist = np.linalg.norm(X - X[:,None], axis = -1)
    result = dist**(alpha)
    return result
    
def calculate_gram_matrix(D):
	rows,cols = D.shape
	gram_matrix = np.empty([rows,cols])
	Di = np.sum(D, axis = 1)
	n = rows
	Disum = np.sum(Di)
	for i in range(rows):
		for j in range(cols):
			gram_matrix[i][j] = D[i][j]/2 + Di[i]/(2*n) + Di[j]/(2*n) - Disum/(2*(n**2))
	return gram_matrix
	
    
if __name__ == "__main__":
	if len(sys.argv) !=4:
		print("usage :", sys.argv[0], "inputfile outputfile alpha")
		sys.exit()
	else:
		alpha = float(sys.argv[3])
		X = np.genfromtxt(sys.argv[1], delimiter =',', dtype='float')
		distance_matrix =  build_distance_matrix(X,alpha)
		distance_matrix_square = np.square(distance_matrix)
		
		#calculate gram matrix
		gram_matrix =  calculate_gram_matrix(distance_matrix_square)
		
		#find evd
		eigvals,eigvecs = np.linalg.eigh(gram_matrix)
		index = np.argsort(eigvals)[:: -1]
		eigvecs = eigvecs[:, index]
		eigvals = eigvals[index]
		k = [index for index,val in enumerate(eigvals) if val<0]
		if len(k)!=0:
			k = k[0]
			eigvals,eigvecs = eigvals[:k], eigvecs[:,:k]
		eigvals_sqrt = np.sqrt(eigvals)
		Y = eigvals_sqrt * eigvecs
		
		with open(sys.argv[2], mode = 'w', newline='') as output:
			np.savetxt(output, Y , newline = " ")