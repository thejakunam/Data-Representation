import sys
import numpy as nump
import warnings

if len(sys.argv) < 3:
    raise Exception("Argument mismatch exception : qerror.py datafile labelsfile")
elif len(sys.argv) >3:
    warnings.warn("Extra Argument warning:Extra Arguements provided!!! \n ignoring all from fourth argument")    
X = nump.genfromtxt(sys.argv[1],delimiter=',',autostrip=True)
#print(X)
assert(X.shape[1] >= 2)  # X should be 2D. Take its first 2 columns
x1=[]
x2=[]
for t in X:
    x1.append(t[0]) # first column of X
    x2.append(t[1]) # second column of X
print('x1={}'.format(x1))
print('x2={}'.format(x2))
Y = nump.genfromtxt(sys.argv[2],autostrip=True)


(unique, counts) = nump.unique(Y, return_counts=True)

udict = {}

i=0
while i<len(X):
    j=0
    while j<len(unique):
        if unique[j] not in udict.keys():
            udict[unique[j]] = 0
        if Y[i] == unique[j]:
            udict[unique[j]] = udict[unique[j]] + X[i]
        j+=1
    i+=1
i=0
while i<len(unique):
    udict[unique[i]] = udict[unique[i]] / counts[i]
    i+=1
e = 0
i=0
while i<len(X):
    j=0
    while j<len(unique):
        if Y[i] == unique[j]:
            e = e + (X[i]-udict[unique[j]])@(X[i]-udict[unique[j]]).T
        j+=1
    i+=1
print(e)