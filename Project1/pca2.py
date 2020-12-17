import sys
import numpy as np

if __name__ == '__main__':
    if len(sys.argv)!=3:
        print('usage : ', sys.argv[0],'inputfile outputfile')
        sys.exit()
        
    df = np.genfromtxt(sys.argv[1], delimiter =",")
    df_mean = np.mean(df)
    df_standard = df - df_mean
    df_covariance = np.matmul(df.T, df)
    eigvals, eigvecs = np.linalg.eig(df_covariance)
    index = np.argsort(eigvals)[::-1]
    eigvals = eigvals[index]
    eigvecs = eigvecs[:,index]
    W = eigvecs.T@df.T
    
    with open(sys.argv[2], mode ='w', newline='') as output:
        np.savetxt(output, W[:,0], newline=" ")
        output.write("\n=======================\n")
        np.savetxt(output, W[:,1], newline=" ")
        