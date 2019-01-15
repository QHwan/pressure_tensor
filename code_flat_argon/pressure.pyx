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
	cdef float *iVec
	cdef float *jVec
	cdef float *ijVec
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
	apm = 1
	nMol /= apm

	Lx = boxMat[0][0][0]
	Ly = boxMat[0][1][1]
	Lz = boxMat[0][2][2]

	Lx2 = Lx*0.5
	Ly2 = Ly*0.5
	Lz2 = Lz*0.5

	pref = 1.6602	# kJ/mol/nm3 -> MPa
	T = 90.
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
	
	iVec = <float *> malloc(3 * sizeof(float))
	jVec = <float *> malloc(3 * sizeof(float))
	ijVec = <float *> malloc(3 * sizeof(float))
	for i in range (3):
		iVec[i] = 0
		jVec[i] = 0
		ijVec[i] = 0

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
			comx += coordMat[apm*i,0]
			comy += coordMat[apm*i,1]
			comz += coordMat[apm*i,2]
		comx /= nMol
		comy /= nMol
		comz /= nMol
		for i in range (apm*nMol):
			coordMat[i,0] -= comx
			coordMat[i,1] -= comy
			coordMat[i,2] -= comz

		# Calculate intermolecular force
		for i in range (nMol):
			for j in range (i+1,nMol):
				for l in range (3):
					iVec[l] = coordMat[i,l]
					jVec[l] = coordMat[j,l]

				if jVec[0]-iVec[0] > Lx2:
					jVec[0] -= Lx
				elif jVec[0]-iVec[0] < -Lx2:
					jVec[0] += Lx
				if jVec[1]-iVec[1] > Ly2:
					jVec[1] -= Ly
				elif jVec[1]-iVec[1] < -Ly2:
					jVec[1] += Ly
				if jVec[2]-iVec[2] > Lz2:
					jVec[2] -= Lz
				elif jVec[2]-iVec[2] < -Lz2:
					jVec[2] += Lz

				ijVec[0] = jVec[0] - iVec[0]
				ijVec[1] = jVec[1] - iVec[1]
				ijVec[2] = jVec[2] - iVec[2]

				rij = sqrt(ijVec[0]**2 + ijVec[1]**2 + ijVec[2]**2)

				if rij > cut:
					fVec[0] = 0
					fVec[1] = 0
					fVec[2] = 0
				else:
					cal_force(fVec, ijVec, rij)

				fMat[i,j,0] = fVec[0]
				fMat[i,j,1] = fVec[1]
				fMat[i,j,2] = fVec[2]

				'''
				if i==1:
					fx += fMat[i,j,0]
					fy += fMat[i,j,1]
					fz += fMat[i,j,2]
		print fx, fy, fz
		'''



		'''
		for i in range (nMol):
			for j in range (nMol):
				if i>j:
					for l in range (3):
						fMat[i,j,l] = -fMat[j,i,l]
		'''




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

				oijVec[0] = ojVec[0] - coordMat[apm*i,0]
				oijVec[1] = ojVec[1] - coordMat[apm*i,1]
				oijVec[2] = ojVec[2] - coordMat[apm*i,2]

				rij = sqrt(oijVec[0]**2 + oijVec[1]**2 + oijVec[2]**2)

				# Check whether force contour passes 
				if rij > cut:
					pass
				else:
					kMin =  int((min(oiVec[2], ojVec[2]) - (zMin - 0.5*dz))/dz)
					kMax =  int((max(oiVec[2], ojVec[2]) - (zMin - 0.5*dz))/dz) 
					for k in range (kMin,kMax+1):

						if oijVec[2] == 0:
							pass
						elif ((zVec[k]-oiVec[2])/oijVec[2])<0 or ((ojVec[2]-zVec[k])/oijVec[2])<0:
							pass
						else:
							ptVec[k] += oijVec[0] * fMat[i,j,0] / fabs(oijVec[2])
							pnVec[k] += oijVec[2] * fMat[i,j,2] / fabs(oijVec[2])

		#print tVec[frame-1], pnVec[400]

	# prepare pK
	for i in range (len(zVec)):
		pkVec[i] = pref*kBT*dVec[i]
	
	for i in range (len(zVec)):
		pnVec[i] *= pref/nFrame_used/vVec[i]
		ptVec[i] *= pref/nFrame_used/vVec[i]
	
		
	print 'nFrame_used = {}'.format(nFrame_used)
	
	# coordt loop is over
	oMat = []
	print '1'
	oMat.append(zVec)
	print '2'
	oMat.append(pkVec)
	print '3'
	oMat.append(ptVec)
	print '4'
	oMat.append(pnVec)
	print '5'
	oMat.append(pkVec+ptVec+pnVec)
	print '6'
	oMat = np.transpose(oMat)
	print '7'

	
	# Save file
	np.savetxt(oFileName,oMat,fmt='%7f')
	print '8'


	# Free everything
	free(iVec)
	free(jVec)
	free(ijVec)

	free(oiVec)
	free(ojVec)

	free(fVec)


		


							






