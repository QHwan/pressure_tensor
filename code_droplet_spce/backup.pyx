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

from func cimport cal_force, cal_pn, cal_pt


@cython.cdivision(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def main():

	# This line we declare the variables
	cdef int i, j, k, l, m
	cdef int frame, nFrame, nFrame_used, nMol, apm
	cdef int lenCoordVec
	cdef float kBT, T
	cdef float doh, dom
	cdef float t
	cdef float comx, comy, comz
	cdef float L, L2
	cdef float rMin, rMax, dr
	cdef float x, y, z, x0, y0, z0
	cdef float fij, ri, rj, rij, ri2, rj2, rij2, ririj
	cdef float Din, Dout, sDin, sDout, Rin, Rout, Rin2, Rout2
	cdef float linn, linp, loutn, loutp, ll
	cdef float la, lb, lap, lbp
	cdef float pN, pT, pK
	cdef float v
	cdef float pref
	cdef float cut

	cdef np.ndarray[float, ndim=3] coordtMat
	cdef np.ndarray[float, ndim=2] coordMat
	cdef np.ndarray[float, ndim=1] tVec
	cdef np.ndarray[float, ndim=1] rVec
	cdef np.ndarray[float, ndim=3] boxMat
	cdef np.ndarray[float, ndim=1] pkVec
	cdef np.ndarray[float, ndim=1] pnVec
	cdef np.ndarray[float, ndim=1] ptVec
	cdef np.ndarray[float, ndim=1] vVec

	cdef np.ndarray[float, ndim=3] fMat

	cdef np.ndarray[float, ndim=1] fLJVec
	cdef np.ndarray[float, ndim=1] fCVec
	fLJVec = np.zeros(3, dtype=np.float32)
	fCVec = np.zeros(3, dtype=np.float32)

#	cdef np.ndarray[float, ndim=2] mMat
#	cdef np.ndarray[float, ndim=1] mVec

	cdef float **iMat
	cdef float **jMat
	cdef float *iVec
	cdef float *jVec
	cdef float *oiVec
	cdef float *ojVec
	cdef float *oijVec
	cdef float *fVec

	cdef int count
	count = 0

	

	# Program starts here
	# Setting the File variable
	trrFileName = sys.argv[1]
	mFileName = sys.argv[2]
	denFileName = sys.argv[3]
	oFileName = sys.argv[4]

	mMat = np.transpose(np.loadtxt(mFileName))
	mVec = mMat[1]

	dMat = np.transpose(np.loadtxt(denFileName, dtype=np.float32))
	rVec = dMat[0]
	dVec = dMat[1]

	with TRRTrajectoryFile(trrFileName) as trrFile:
		coordtMat, tVec, step, boxMat, lambdaVec = trrFile.read()
	trrFile.close()

	nFrame, nMol, xyz = np.shape(coordtMat)
	apm = 3
	nMol /= apm

	L = boxMat[0][0][0]
	L2 = L*0.5

	pref = 1.6602	# kJ/mol/nm3 -> MPa
	T = 300.
	kBT = 2.479*T/298

	cut = 4.0

	# Setting PMat
	dr = rVec[1]-rVec[0]
	rMin = dr*0.5
	rMax = rVec[len(rVec)-1]

	pkVec = np.zeros( (len(rVec)), dtype=np.float32)
	pnVec = np.zeros( (len(rVec)), dtype=np.float32)
	ptVec = np.zeros( (len(rVec)), dtype=np.float32)



	vVec = np.zeros(len(rVec), dtype=np.float32)
	for i in range (len(rVec)):
	#	vVec[i] = 4*math.pi/3.*(pow(rVec[i] + dr*0.5,3) - pow(rVec[i] - dr*0.5,3))
		vVec[i] = 4*math.pi*rVec[i]*rVec[i]


	fMat = np.zeros( (nMol,nMol,3), dtype=np.float32)
	fLJMat = np.zeros( (nMol, nMol, 3), dtype=np.float32)
	fCMat = np.zeros( (nMol, nMol, 3), dtype=np.float32)

	if len(rVec) != len(dVec):
		print 'rVec and denVec have different length'
		#exit(1)

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

	
	iVec = <float *> malloc(3 * sizeof(float))
	jVec = <float *> malloc(3 * sizeof(float))
	for i in range (3):
		iVec[i] = 0
		jVec[i] = 0

	oiVec = <float *> malloc(3 * sizeof(float))
	ojVec = <float *> malloc(3 * sizeof(float))
	oijVec = <float *> malloc(3 * sizeof(float))
	for i in range (3):
		oiVec[i] = 0
		ojVec[i] = 0
		oijVec[i] =0 

	fVec = <float *> malloc(3 * sizeof(float))
	for i in range (3):
		fVec[i] = 0


	
	# Starting read coordinates
	frame = 0
	nFrame_used = 0
	for coordMat in coordtMat:

		if frame == 10000:
			break

		if frame % 100 == 0:
			print frame

		if mVec[frame] != nMol and nFrame != 1:
			frame += 1
			pass

		else:
			frame += 1
			nFrame_used += 1


			'''
			# Consider periodic boundary
			x0 = coordMat[0,0]; y0 = coordMat[0,1]; z0 = coordMat[0,2]
			for i in range (1, nMol):
				x = coordMat[apm*i,0]
				y = coordMat[apm*i,1]
				z = coordMat[apm*i,2]
				if x-x0 > L2:
					for j in range (apm):
						coordMat[apm*i+j,0] -= L
				elif x-x0 < -L2:
					for j in range (apm):
						coordMat[apm*i+j,0] += L
				if y-y0 > L2:
					for j in range (apm):
						coordMat[apm*i+j,1] -= L
				elif y-y0 < -L2:
					for j in range (apm):
						coordMat[apm*i+j,1] += L
				if z-z0 > L2:
					for j in range (apm):
						coordMat[apm*i+j,2] -= L
				elif z-z0 < -L2:
					for j in range (apm):
						coordMat[apm*i+j,2] += L
			'''


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
					for k in range (apm):
						iMat[k][0] = coordMat[apm*i+k,0]
						iMat[k][1] = coordMat[apm*i+k,1]
						iMat[k][2] = coordMat[apm*i+k,2]
						jMat[k][0] = coordMat[apm*j+k,0]
						jMat[k][1] = coordMat[apm*j+k,1]
						jMat[k][2] = coordMat[apm*j+k,2]

					'''
					if jMat[0][0]-iMat[0][0] > L2:
						for k in range (apm):
							jMat[k][0] -= L

					elif jMat[0][0]-iMat[0][0] < -L2:
						for k in range (apm):
							jMat[k][0] += L

					if jMat[0][1]-iMat[0][1] > L2:
						for k in range (apm):
							jMat[k][1] -= L

					elif jMat[0][1]-iMat[0][1] < -L2:
						for k in range (apm):
							jMat[k][1] += L

					if jMat[0][2]-iMat[0][2] > L2:
						for k in range (apm):
							jMat[k][2] -= L

					elif jMat[0][2]-iMat[0][2] < -L2:
						for k in range (apm):
							jMat[k][2] += L
					'''

					fVec[0] = 0
					fVec[1] = 0
					fVec[2] = 0
					cal_force(fVec, iMat, jMat, cut)

					fMat[i,j,0] = fVec[0]
					fMat[i,j,1] = fVec[1]
					fMat[i,j,2] = fVec[2]


			'''
			for i in range (nMol):
				for j in range (nMol):
					if i>j:
						for l in range (3):
							fMat[i,j,l] = -fMat[j,i,l]
							fLJMat[i,j,l] = -fLJMat[j,i,l]
							fCMat[i,j,l] = -fCMat[j,i,l]
			'''




			# Calculate Pressure
			for i in range (nMol):

				oiVec[0] = coordMat[apm*i,0]
				oiVec[1] = coordMat[apm*i,1]
				oiVec[2] = coordMat[apm*i,2]

				ri2 = oiVec[0]**2 + oiVec[1]**2 + oiVec[2]**2

				for j in range (i+1, nMol):
					ojVec[0] = coordMat[apm*j,0]
					ojVec[1] = coordMat[apm*j,1]
					ojVec[2] = coordMat[apm*j,2]

					if ojVec[0]-oiVec[0] > L2:
						ojVec[0] -= L
					elif ojVec[0]-oiVec[0] < -L2:
						ojVec[0] += L
					if ojVec[1]-oiVec[1] > L2:
						ojVec[1] -= L
					elif ojVec[1]-oiVec[1] < -L2:
						ojVec[1] += L
					if ojVec[2]-oiVec[2] > L2:
						ojVec[2] -= L
					elif ojVec[2]-oiVec[2] < -L2:
						ojVec[2] += L

					oijVec[0] = ojVec[0] - oiVec[0]
					oijVec[1] = ojVec[1] - oiVec[1]
					oijVec[2] = ojVec[2] - oiVec[2]


					rij2 = oijVec[0]**2 + oijVec[1]**2 + oijVec[2]**2
					rij = sqrt(rij2)
					ririj = oiVec[0]*oijVec[0] + oiVec[1]*oijVec[1] + oiVec[2]*oijVec[2]

					for k in range (len(rVec)):
						r = rVec[k]

						# Check D
						D = ririj**2 - rij2*(ri2 - r*r)

						if D <= 0:
							pass
						else:
							lp = -ririj/rij2 + sqrt(D)/rij2
							ln = -ririj/rij2 - sqrt(D)/rij2
							
							fVec[0] = fMat[i,j,0]
							fVec[1] = fMat[i,j,1]
							fVec[2] = fMat[i,j,2]

							#fij = sqrt(fVec[0]**2 + fVec[1]**2 + fVec[2]**2)
							fij = (fVec[0]*oijVec[0] + fVec[1]*oijVec[1] + fVec[2]*oijVec[2])/rij

							if ln >= 0 and ln <= 1:
								pnVec[k] += fabs(cal_pn(fVec, oiVec, oijVec, ln))*fij/rij
								ptVec[k] += fabs(cal_pt(fVec, oiVec, oijVec, ln))*fij/rij

							if lp >= 0 and lp <= 1:
								pnVec[k] += fabs(cal_pn(fVec, oiVec, oijVec, lp))*fij/rij
								ptVec[k] += fabs(cal_pt(fVec, oiVec, oijVec, lp))*fij/rij
								


	# prepare pK
	for i in range (len(rVec)):
		pkVec[i] = pref*kBT*dVec[i]
	
	for i in range (len(rVec)):
		pnVec[i] *= pref/nFrame_used/vVec[i]
		ptVec[i] *= pref/nFrame_used/vVec[i]
	
		
	print 'nFrame_used = {}'.format(nFrame_used)
	
	# coordt loop is over
	oMat = []
	oMat.append(rVec)
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

	free(iVec)
	free(jVec)

	free(oiVec)
	free(ojVec)
	free(oijVec)

	free(fVec)

	print count

		


							






