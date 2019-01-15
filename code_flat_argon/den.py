import numpy as np
import sys
import math
from mdtraj.formats import XTCTrajectoryFile
from mdtraj.formats import TRRTrajectoryFile

trrFileName = sys.argv[1]
oFileName = sys.argv[2]


with TRRTrajectoryFile(trrFileName) as trrFile:
	coordtMat, tVec, step, boxMat, lambdaVec = trrFile.read()
trrFile.close()

nFrame, mMol, xyz = np.shape(coordtMat)
print 'nFrame = {}'.format(nFrame)
apm = 1
mMol /= apm

Lx = boxMat[0][0][0]
Ly = boxMat[0][1][1]
Lz = boxMat[0][2][2]
Lx2 = Lx*0.5
Ly2 = Ly*0.5
Lz2 = Lz*0.5

# Setting PMat
dz = 0.02
zMin = -8. + dz*0.5
zMax = 8. - dz*0.5
zVec = np.zeros(int((zMax-zMin)/dz)+1)
denVec = np.zeros(len(zVec))


for i in range (len(zVec)):
	zVec[i] = zMin + dz*i

vVec = np.zeros(len(zVec))
for i in range (len(zVec)):
	vVec[i] = Lx*Ly*dz

frame = 0
nFrame_used = 0
for coordMat in coordtMat:
	if frame % 100 == 0:
		print frame


	frame += 1
	nFrame_used += 1

	# Consider periodic boundary
	x0 = coordMat[0,0]; y0 = coordMat[0,1]; z0 = coordMat[0,2]
	for i in range (1, mMol):
		x = coordMat[apm*i,0]
		y = coordMat[apm*i,1]
		z = coordMat[apm*i,2]
		if x-x0 > Lx2:
			for j in range (apm):
				coordMat[apm*i+j,0] -= Lx
		elif x-x0 < -Lx2:
			for j in range (apm):
				coordMat[apm*i+j,0] += Lx
		if y-y0 > Ly2:
			for j in range (apm):
				coordMat[apm*i+j,1] -= Ly
		elif y-y0 < -Ly2:
			for j in range (apm):
				coordMat[apm*i+j,1] += Ly
		if z-z0 > Lz2:
			for j in range (apm):
				coordMat[apm*i+j,2] -= Lz
		elif z-z0 < -Lz2:
			for j in range (apm):
				coordMat[apm*i+j,2] += Lz

	# calculate com matrix using OW position
	comx = 0; comy = 0; comz = 0;
	for i in range (mMol):
		comz += coordMat[apm*i,2]
	comz /= mMol
	for i in range (apm*mMol):
		coordMat[i,2] -= comz


	for i in range (mMol):
		z = coordMat[apm*i,2]

		if z>=-8 and z<8:
			denVec[int((z+8)/dz)] += 1
		else:
			pass


for i in range (len(zVec)):
	denVec[i] *= 1/vVec[i]/nFrame_used

oMat = []
oMat.append(zVec)
oMat.append(denVec)
oMat = np.transpose(oMat)

np.savetxt(oFileName,oMat,fmt='%7f')
