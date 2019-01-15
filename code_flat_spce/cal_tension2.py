import numpy as np
import sys
import scipy.integrate
import scipy.signal

iFileName = sys.argv[1]
oFileName = sys.argv[2]

iMat = np.transpose(np.loadtxt(iFileName))
rVec = iMat[0]
pkVec = iMat[1]
uVec = iMat[2]
ptVec = iMat[3]

rMax = rVec[len(rVec)-1]


pnVec = np.zeros(len(rVec))

oMat = []
yVec = np.zeros(len(rVec))
xVec = np.zeros(len(rVec))
bufVec = np.zeros(len(rVec))
buf2Vec = np.zeros(len(rVec))
ptVec_new = np.zeros(len(rVec))


# Smoothing curve
#ptVec_new = scipy.signal.savgol_filter(ptVec,window_length=11,polyorder=3)


# Correcting value
'''
for i in range (len(rVec)):
	yVec[i] = ptVec_new[i]*rVec[i]
correction = np.trapz(yVec,rVec)*2/rMax/rMax
print correction
'''


pnVec[0] = ptVec[0]

for i in range (1,len(rVec)):
	yVec = np.zeros(i+1)
	xVec = np.zeros(i+1)

	for j in range (len(xVec)):
		xVec[j] = rVec[j]
		yVec[j] = rVec[j]*rVec[j]*(uVec[j] + 3*pkVec[j])

	pnVec[i] = scipy.integrate.simps(yVec,xVec) * 1./rVec[i]/rVec[i]/rVec[i]

for i in range (len(rVec)):
	ptVec[i] = uVec[i] + 3*pkVec[i] - pnVec[i]
	ptVec[i] *= 0.5


#pnVec[0] -= correction

# Calculation tension
yVec = np.zeros(len(rVec))
for i in range (1, len(rVec)):
	yVec[i] = pnVec[i] - ptVec[i]
	bufVec[i] = yVec[i] 
print np.trapz(yVec,rVec)




oMat.append(rVec)
oMat.append(ptVec)
oMat.append(pnVec)
oMat = np.transpose(oMat)

np.savetxt(oFileName,oMat,fmt='%7f')

