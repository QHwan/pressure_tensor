import cython
cimport cython
import numpy as np
cimport numpy as np
import sys
import math
from libc.math cimport sqrt, fabs
from libc.stdlib cimport malloc, free
import time
from mdtraj.formats import XTCTrajectoryFile
from mdtraj.formats import TRRTrajectoryFile

from func cimport cal_force


@cython.cdivision(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def main():

	# This line we declare the variables
	cdef int i, j, k, l, m
	cdef int kMin, kMax
	cdef int frame, nFrame, nFrame_used, nMol, apm
	cdef int lenZVec
	cdef float kBT, T
	cdef float t
	cdef float comx, comy, comz
	cdef float Lx, Ly, Lz
	cdef float Lx2, Ly2, Lz2
	cdef float zMin, zMax, dz
	cdef float x, y, z, x0, y0, z0
	cdef float zi, zj, zij
	cdef float ri, rj, rij
	cdef float pN, pT, pK
	cdef float v
	cdef float pref
	cdef float cut
	cdef float fx, fy, fz

	cdef np.ndarray[float, ndim=3] coordtMat
	cdef np.ndarray[float, ndim=2] coordMat
	cdef np.ndarray[float, ndim=1] tVec
	cdef np.ndarray[float, ndim=1] zVec
	cdef np.ndarray[float, ndim=3] boxMat
	cdef np.ndarray[float, ndim=1] pkVec
	cdef np.ndarray[float, ndim=1] pnVec
	cdef np.ndarray[float, ndim=1] ptVec
	cdef np.ndarray[float, ndim=1] vVec

	cdef np.ndarray[float, ndim=3] fMat


#	cdef np.ndarray[float, ndim=2] mMat
#	cdef np.ndarray[float, ndim=1] mVec

	cdef np.ndarray[float, ndim=1] oijVec
	cdef float **iMat
	cdef float **jMat
	cdef float *oiVec
	cdef float *ojVec
	cdef float *fVec

	cdef int count
	count = 0

	

	# Program starts here
	# Setting the File variable
	trrFileName = sys.argv[1]
	denFileName = sys.argv[2]
	oFileName = sys.argv[3]

	dMat = np.transpose(np.loadtxt(denFileName, dtype=np.float32))
	zVec = dMat[0]
	dVec = dMat[1]

	lenZVec = len(zVec)

	with TRRTrajectoryFile(trrFileName) as trrFile:
		coordtMat, tVec, step, boxMat, lambdaVec = trrFile.read()
	trrFile.close()

	nFrame, nMol, xyz = np.shape(coordtMat)
	apm = 3
	nMol /= apm

	Lx = boxMat[0][0][0]
	Ly = boxMat[0][1][1]
	Lz = boxMat[0][2][2]

	Lx2 = Lx*0.5
	Ly2 = Ly*0.5
	Lz2 = Lz*0.5

	pref = 1.6602	# kJ/mol/nm3 -> MPa
	T = 300.
	kBT = 2.479*T/298

	cut = 1.2

	# Setting PMat
	dz = zVec[1]-zVec[0]
	zMin = zVec[0]
	zMax = zVec[len(zVec)-1]

	pkVec = np.zeros( (len(zVec)), dtype=np.float32)
	pnVec = np.zeros( (len(zVec)), dtype=np.float32)
	ptVec = np.zeros( (len(zVec)), dtype=np.float32)

	vVec = np.zeros(len(zVec), dtype=np.float32)
	for i in range (len(zVec)):
		vVec[i] = Lx*Ly*1


	fMat = np.zeros( (nMol,nMol,3), dtype=np.float32)

	if len(zVec) != len(dVec):
		print 'zVec and denVec have different length'


	# Setting iVec and jVec
	iMat = <float **> malloc(apm * sizeof(float*))
	jMat = <float **> malloc(apm * sizeof(float*))
	for i in range (apm):
		iMat[i] = <float *> malloc(3 * sizeof(float))
		jMat[i] = <float *> malloc(3 * sizeof(float))
	for i in range (apm):
		for j in range (3):
			iMat[i][j] = 0
			jMat[i][j] = 0

	oiVec = <float *> malloc(3 * sizeof(float))
	ojVec = <float *> malloc(3 * sizeof(float))
	oijVec = np.zeros(3, dtype=np.float32)
	for i in range (3):
		oiVec[i] = 0
		ojVec[i]  = 0

	fVec = <float *> malloc(3 * sizeof(float))
	for i in range (3):
		fVec[i] = 0


	
	# Starting read coordinates
	frame = 0
	nFrame_used = 0
	for coordMat in coordtMat:

		#if frame == 100:
		#	break

		frame += 1
		nFrame_used += 1

		# Consider periodic boundary
		x0 = coordMat[0,0]; y0 = coordMat[0,1]; z0 = coordMat[0,2]
		for i in range (1, nMol):
			x = coordMat[apm*i,0]
			y = coordMat[apm*i,1]
			z = coordMat[apm*i,2]
			if z-z0 > Lz2:
				for j in range (apm):
					coordMat[apm*i+j,2] -= Lz
			elif z-z0 < -Lz2:
				for j in range (apm):
					coordMat[apm*i+j,2] += Lz

		# calculate com matrix using OW position
		comx = 0; comy = 0; comz = 0;
		for i in range (nMol):
			comz += coordMat[apm*i,2]
		comz /= nMol
		for i in range (apm*nMol):
			coordMat[i,2] -= comz

		# Calculate intermolecular force
		for i in range (nMol):
			for j in range (i+1,nMol):
				for k in range (apm):
					iMat[k][0] = coordMat[apm*i+k,0]
					iMat[k][1] = coordMat[apm*i+k,1]
					iMat[k][2] = coordMat[apm*i+k,2]
					jMat[k][0] = coordMat[apm*j+k,0]
					jMat[k][1] = coordMat[apm*j+k,1]
					jMat[k][2] = coordMat[apm*j+k,2]

				if jMat[0][0]-iMat[0][0] > Lx2:
					for k in range (apm):
						jMat[k][0] -= Lx

				elif jMat[0][0]-iMat[0][0] < -Lx2:
					for k in range (apm):
						jMat[k][0] += Lx

				if jMat[0][1]-iMat[0][1] > Ly2:
					for k in range (apm):
						jMat[k][1] -= Ly

				elif jMat[0][1]-iMat[0][1] < -Ly2:
					for k in range (apm):
						jMat[k][1] += Ly

				if jMat[0][2]-iMat[0][2] > Lz2:
					for k in range (apm):
						jMat[k][2] -= Lz

				elif jMat[0][2]-iMat[0][2] < -Lz2:
					for k in range (apm):
						jMat[k][2] += Lz

				fVec[0] = 0
				fVec[1] = 0
				fVec[2] = 0

				cal_force(fVec, iMat, jMat)

				fMat[i,j,0] = fVec[0]
				fMat[i,j,1] = fVec[1]
				fMat[i,j,2] = fVec[2]

				



		# Calculate Pressure
		for i in range (nMol):

			oiVec[0] = coordMat[apm*i,0]
			oiVec[1] = coordMat[apm*i,1]
			oiVec[2] = coordMat[apm*i,2]

			for j in range (i+1, nMol):

				ojVec[0] = coordMat[apm*j,0]
				ojVec[1] = coordMat[apm*j,1]
				ojVec[2] = coordMat[apm*j,2]

				if ojVec[0]-oiVec[0] > Lx2:
					ojVec[0] -= Lx
				elif ojVec[0]-oiVec[0] < -Lx2:
					ojVec[0] += Lx
				if ojVec[1]-oiVec[1] > Ly2:
					ojVec[1] -= Ly
				elif ojVec[1]-oiVec[1] < -Ly2:
					ojVec[1] += Ly
				if ojVec[2]-oiVec[2] > Lz2:
					ojVec[2] -= Lz
				elif ojVec[2]-oiVec[2] < -Lz2:
					ojVec[2] += Lz

				oijVec[0] = ojVec[0] - oiVec[0]
				oijVec[1] = ojVec[1] - oiVec[1]
				oijVec[2] = ojVec[2] - oiVec[2]


				# Check whether force contour passes 
				kMin = int((min(oiVec[2],ojVec[2]) - (zMin-0.5*dz))/dz)
				kMax = int((max(oiVec[2],ojVec[2]) - (zMin-0.5*dz))/dz) 
				for k in range (kMin,kMax+1):
					if oijVec[2] == 0:
						pass
					elif ((zVec[k]-oiVec[2])/oijVec[2])<0 or ((ojVec[2]-zVec[k])/oijVec[2])<0:
						pass
					else:
						ptVec[k] += oijVec[0] * fMat[i,j,0] / fabs(oijVec[2])
						pnVec[k] += oijVec[2] * fMat[i,j,2] / fabs(oijVec[2])


	# prepare pK
	for i in range (len(zVec)):
		pkVec[i] = pref*kBT*dVec[i]
	
	print nFrame_used
	for i in range (len(zVec)):
		pnVec[i] *= pref/nFrame_used/vVec[i]
		ptVec[i] *= pref/nFrame_used/vVec[i]
	
		
	print 'nFrame_used = {}'.format(nFrame_used)
	
	# coordt loop is over
	oMat = []
	oMat.append(zVec)
	oMat.append(pkVec)
	oMat.append(ptVec)
	oMat.append(pnVec)
	oMat.append(pkVec+ptVec+pnVec)
	oMat = np.transpose(oMat)

	
	# Save file
	np.savetxt(oFileName,oMat,fmt='%7f')


	# Free everything
	for i in range (apm):
		free(iMat[i])
		free(jMat[i])
	free(iMat)
	free(jMat)


	free(oiVec)
	free(ojVec)

	free(fVec)

