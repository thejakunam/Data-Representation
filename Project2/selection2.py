import numpy as np
from scipy.sparse import *
from skfeature.utility.construct_W import construct_W
import sys


def fisher_score(X, y):
    W = construct_W(X)

    D = np.array(W.sum(axis=1))
    L = W
    tmp = np.dot(np.transpose(D), X)
    D = diags(np.transpose(D), [0])
    Xt = np.transpose(X)
    t1 = np.transpose(np.dot(Xt, D.todense()))
    t2 = np.transpose(np.dot(Xt, L.todense()))
   
    D_prime = np.sum(np.multiply(t1, X), 0) - np.multiply(tmp, tmp)/D.sum()
    
    L_prime = np.sum(np.multiply(t2, X), 0) - np.multiply(tmp, tmp)/D.sum()
   
    D_prime[D_prime < 1e-12] = 10000
    lap_score = 1 - np.array(np.multiply(L_prime, 1/D_prime))[0, :]

  
    score = 1.0/lap_score - 1
    return np.transpose(score)


def feature_ranking(score):
    idx = np.argsort(score, 0)
    return idx[::-1]
	
if __name__ == "__main__":
	if len(sys.argv) !=4:
		print("usage :", sys.argv[0], "data label output")
		sys.exit()
	else:
		X = np.genfromtxt(sys.argv[1], delimiter =',', dtype='float')
		y = np.genfromtxt(sys.argv[2], delimiter =',', dtype='float')
		score = fisher_score(X,y)
		print(score)
		rank = feature_ranking(score)
		print(rank)
		with open(sys.argv[3], mode = 'w', newline='') as output:
			np.savetxt(output, score , newline = " ")
			output.write("\n==========================\n")
			np.savetxt(output, rank.astype(int) , newline = " ", fmt='%5.0f')
		