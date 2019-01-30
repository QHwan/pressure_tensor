import numpy as np
import sys

i1FileName = 'p360_lj.xvg'
i2FileName = 'p360_c.xvg'

i1Mat = np.loadtxt(i1FileName)
i2Mat = np.loadtxt(i2FileName)

rVec = i1Mat[:,0]
ljVec = i1Mat[:,3]
cVec = i2Mat[:,3]

oMat = []
oMat.append(rVec)
oMat.append(ljVec)
oMat.append(cVec)
oMat.append(ljVec+cVec)

np.savetxt('p.xvg',np.transpose(oMat),fmt='%5f')
