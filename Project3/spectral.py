import sys
import random
import numpy as np
import math
import kmeanspp
random.seed(1)
np.random.seed(1)

class spectral_clustering:
    @staticmethod
    def computePairDistances(rows,cols,data):
        pairDistances = np.zeros((rows, rows))
        for i in range(0, rows):
            for j in range(0, rows):
                distance = 0
                for col in range(0, cols):
                    distance += math.pow((data[i][col]-data[j][col]),2)
                pairDistances[i][j] = math.sqrt(distance)
        return pairDistances

    @staticmethod
    def computeWMatrix(rows,pairDistances,sigma ):
        W = np.zeros((rows,rows))
        for i in range(0,rows):
            for j in range(0,rows):
                W[i][j] = math.exp(-math.pow((pairDistances[i][j]),2)/(2*math.pow(sigma,2)))
        return W

    @staticmethod
    def computeDMatrix(rows,W):
        D = np.zeros((rows,rows))
        for i in range(0,rows):
            wSum = 0
            for j in range(0,rows):
                wSum += W[i][j]
            D[i][i] = wSum
        return D

    @staticmethod
    def computeLMatrix(rows,D,W):
        L = np.zeros((rows,rows))
        for i in range(0, rows):
            for j in range(0, rows):
                if(i==j and D[i][i] != 0):
                    L[i][j] = 1
                elif(i!=j and D[i][i]!=0 and D[j][j]!=0):
                    L[i][j] = -W[i][j]/(math.sqrt(D[i][i]*D[j][j]))
                else:
                    L[i][j]=0
        return L

if __name__=="__main__":
    if len(sys.argv) != 5:
        print ("usage : spectral.py inputfile kval sigmaval outputfile")
        sys.exit(0)
    else:
        data=np.genfromtxt(sys.argv[1], delimiter = ",", dtype = float)
        rows = data.shape[0]
        cols = data.shape[1]
        k = int(sys.argv[2])
        sigmaval = float(sys.argv[3])
        pairDistances = spectral_clustering.computePairDistances(rows,cols,data)
        W = spectral_clustering.computeWMatrix(rows,pairDistances,sigmaval )
        D = spectral_clustering.computeDMatrix(rows,W)
        L = spectral_clustering.computeLMatrix(rows,D,W)
        evals, evecs = np.linalg.eigh(L)
        sorted_idx = np.argsort(evals)
        sorted_evals = evals[sorted_idx]
        sorted_evecs = evecs[:, sorted_idx]
        k_vecs = sorted_evecs[:,0:k]
        kevecs = np.zeros((len(k_vecs), k))
        for i in range(0,k):
            for j in range(0,len(k_vecs)):
                kevecs[j][i] = k_vecs[j][i]/math.sqrt(D[j][j])
        data = kevecs
        rows = data.shape[0]
        cols = data.shape[1]
        r = 10
        clusters = kmeanspp.kmeansplusplus.computeCluster(data, k, r)
        arr_qe = []
        for c in clusters:
            labels = np.unique(c)
            m = labels.shape[0]
            error = 0
            for k in range(1 , m + 1):
                group = data[np.array(c) == int(k)]
                error += kmeanspp.kmeansplusplus.qerror(group,np.mean(group,0))
            arr_qe.append(error)
        print("Errors for all iterations: \n", arr_qe)
        print("\n Final Quantization Error: \n", min(arr_qe))
        with open(sys.argv[4], mode = 'w', newline='') as op:
            np.savetxt(op,clusters , newline = " ")