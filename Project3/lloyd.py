import warnings
import numpy as nump
import random
import sys
nump.random.seed(1)

class lloyd_kmeans:
    @staticmethod
    def computeCentres(data, centres):
        print("Centres are:")
        print(centres)
        init_clusters = []
        for d in data:
            distances = []
            for c in centres:
                distances.append(nump.sum(nump.square(d-c)))
            init_clusters.append(distances.index(min(distances)))
        print(" Clustering output obtained :")
        print(init_clusters)
        #error calculation
        error = 0
        for e,l in zip(data,init_clusters):
            error+=nump.square(e-centres[l])
        error = (nump.sum(error))
        print("Quantization Error: ")
        print(error)
        output_labels=set(init_clusters)
        data_labels={}
        centres=[]  
        for l in output_labels:
           data_labels[l]=[]
        for e,l in zip(data,init_clusters):
            data_labels[l].append(e)
        for l in output_labels:
            centres.append(sum(data_labels.get(l))/len(data_labels.get(l)))
        return centres

def mainfunc():
    if len(sys.argv) < 5:
        raise Exception("Argument mismatch exception : lloyd.py inputfile kval rval outputfile")
    elif len(sys.argv) >5:
        warnings.warn("Extra Argument warning:Extra Arguements provided!!! \n ignoring all from sixth argument")       
    k=int(sys.argv[2])
    r=int(sys.argv[3])
    with open(sys.argv[1],"rt") as datafile:
        data=nump.loadtxt(datafile, delimiter=",")
    idx= nump.random.choice(data.shape[0], size=k, replace=False)
    centres=data[idx]
    for i in range(r):
        centres=lloyd_kmeans.computeCentres(data,centres)
    labels=[]
    for d in data:
        distances=[]
        for c in centres:
            distances.append(nump.sum(nump.square(d-c)))
        labels.append(distances.index(min(distances)))
    print("Final Labels are : ")
    print(labels)
    print("Quantization error printed on console :")
        #error calculation
    error = 0
    for e,l in zip(data,labels):
        error+=nump.square(e-centres[l])
    error = (nump.sum(error))
    print(error)
    with open(sys.argv[4], mode = 'wt', newline='') as outputfile:
        nump.savetxt(outputfile,labels , newline = " ")

if __name__=="__main__":
    mainfunc()