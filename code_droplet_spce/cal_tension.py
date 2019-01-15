import numpy as np
import sys

iFileName = sys.argv[1]

iMat = np.transpose(np.loadtxt(iFileName))
zVec = iMat[0]
ptVec = iMat[2]
pnVec = iMat[3]

yVec = np.zeros(len(zVec))
for i in range (len(zVec)):
	yVec[i] = pnVec[i]-ptVec[i]


print 0.5*np.trapz(yVec,zVec)

