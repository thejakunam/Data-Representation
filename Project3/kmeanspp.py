import sys
import random
import numpy as np
import math

random.seed(1)
np.random.seed(1)

class kmeansplusplus:
    @staticmethod
    def getLabels(data, means):
        labels = []
        for d in data:
            distances = []
            for m in means:
                distances.append(sum([(a - b) ** 2 for a, b in zip(d , m)]))
            min_dist = distances.index(min(distances))
            labels.append(min_dist + 1)
        return labels

    @staticmethod
    def computeCluster(data, k, r):
        clusters = []
        for i in range(r):
            means = []
            probability = []
            randno = random.randrange(0, data.shape[0])
            means.append(data[randno])
            for j in range(1, k ):
                distances = []
                for x in data:
                    d_centre = []
                    for c in means:
                        d_centre.append(sum([(a - b) ** 2 for a, b in zip(x , c)]))
                    distances.append(min(d_centre)) 
                distanceSum = sum(distances)
                probability = list(distances/distanceSum)
                p_rand = np.random.choice(probability)
                p_idx = probability.index(p_rand)
                means.append(data[p_idx])
            labels = kmeansplusplus.getLabels(data, means)
            new_centers = []
            while new_centers != means:
                new_centers = []
                for i in range(1, len(labels) + 1):
                    group = data[np.array(labels) == i]
                    new_centers.append(np.mean(group, axis = 0))
                means = new_centers
            labels = kmeansplusplus.getLabels(data, means)
            clusters.append(labels)
        return clusters

    @staticmethod
    def qerror(grp,mean):
        err = 0
        for i in range(0, grp.shape[0]):
            err += sum([(a - b) ** 2 for a, b in zip(grp[i], mean)])
        return err


if __name__=="__main__":
    if len(sys.argv) != 5:
        print ("usage : kmeanspp.py inputfile kval rval outputfile")
        sys.exit(0)
    else:
        k=int(sys.argv[2])
        r=int(sys.argv[3])
        data=np.genfromtxt(sys.argv[1], delimiter = ",", dtype = float)
        clusters = kmeansplusplus.computeCluster(data, k, r)
        arr_qe = []
        for c in clusters:
            labels = np.unique(c)
            m = labels.shape[0]
            error = 0
            for k in range(1 , m + 1):
                group = data[np.array(c) == int(k)]
                error += kmeansplusplus.qerror(group,np.mean(group,0))
            arr_qe.append(error)
        print("Errors for all iterations: \n", arr_qe)
        print("\n Final Quantization Error: \n", min(arr_qe))
        with open(sys.argv[4], mode = 'w', newline='') as op:
            np.savetxt(op,clusters , newline = " ")
