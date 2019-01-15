import numpy as np
import sys

iFileName = sys.argv[1]
oFileName = sys.argv[2]

iMat = np.transpose(np.loadtxt(iFileName))
rVec = iMat[0]
pnVec = iMat[3]
ptVec = iMat[5]

oMat = []
yVec = np.zeros(len(rVec))
for i in range (len(rVec)):
	yVec[i] = pnVec[i]-ptVec[i]
oMat.append(rVec)
oMat.append(yVec)
oMat = np.transpose(oMat)

np.savetxt(oFileName,oMat,fmt='%7f')

print np.trapz(yVec,rVec)

