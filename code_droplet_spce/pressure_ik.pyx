import cython
cimport cython
import numpy as np
cimport numpy as np
import sys
import math
from libc.math cimport sqrt, atan, log, pi, fabs
from libc.stdlib cimport malloc, free
from libc.float cimport FLT_EPSILON
import time
from mdtraj.formats import XTCTrajectoryFile
from mdtraj.formats import TRRTrajectoryFile

from func cimport integral_range, cal_force, cal_pn, cal_pt, vec_dot, vec_minus


@cython.cdivision(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def main():


	# This line we declare the variables
	cdef int i, j, k, l, m
	cdef int frame, nFrame, nFrame_used, nMol, apm
	cdef int lenCoordVec
	cdef int kMin
	cdef double kBT, T
	cdef double doh, dom
	cdef double r, f, t, c
	cdef double comx, comy, comz
	cdef double L, L2
	cdef double rMin, rMax, dr
	cdef double x, y, z, x0, y0, z0
	cdef double ri, rj, rij, ri2, rj2, rij2, ririj
	cdef double Din, Dout, sDin, sDout, Rin, Rout, Rin2, Rout2, Da, Db
	cdef double linn, linp, loutn, loutp, ll, l0, l02
	cdef double la, lb, lap, lbp
	cdef double pN, pT, pK
	cdef double v
	cdef double pref
	cdef double cut

	cdef np.ndarray[float, ndim=3] coordtMat
	cdef np.ndarray[float, ndim=2] coordMat
	cdef np.ndarray[float, ndim=1] tVec
	cdef np.ndarray[double, ndim=1] rVec
	cdef np.ndarray[double, ndim=1] Rin2Vec
	cdef np.ndarray[double, ndim=1] Rout2Vec
	cdef np.ndarray[float, ndim=3] boxMat
	cdef np.ndarray[double, ndim=1] pkVec
	cdef np.ndarray[double, ndim=1] pnVec
	cdef np.ndarray[double, ndim=1] ptVec
	cdef np.ndarray[double, ndim=1] vVec

	cdef np.ndarray[double, ndim=3] fMat

#	cdef np.ndarray[double, ndim=2] mMat
#	cdef np.ndarray[double, ndim=1] mVec

	cdef double **iMat
	cdef double **jMat
	cdef double *iVec
	cdef double *jVec
	cdef double *oiVec
	cdef double *ojVec
	cdef double *oijVec
	cdef double *fVec


	

	# Program starts here
	# Setting the File variable
	trrFileName = sys.argv[1]
	mFileName = sys.argv[2]
	denFileName = sys.argv[3]
	oFileName = sys.argv[4]

	mMat = np.transpose(np.loadtxt(mFileName))
	mVec = mMat[1]

	dMat = np.transpose(np.loadtxt(denFileName))
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
	T = 260.
	kBT = 2.479*T/298

	cut = 4.0

	# Setting PMat
	dr = rVec[1]-rVec[0]
	rMin = dr*0.5
	rMax = rVec[len(rVec)-1]

	pkVec = np.zeros( len(rVec))
	pnVec = np.zeros( len(rVec))
	ptVec = np.zeros( len(rVec))


	fMat = np.zeros( (nMol,nMol, 3) )

	if len(rVec) != len(dVec):
		print 'rVec and denVec have different length'
		exit(1)


	# Setting iVec and jVec
	iMat = <double **> malloc(apm * sizeof(double*))
	jMat = <double **> malloc(apm * sizeof(double*))
	for i in range (apm):
		iMat[i] = <double *> malloc(3 * sizeof(double))
		jMat[i] = <double *> malloc(3 * sizeof(double))
	for i in range (apm):
		for j in range (3):
			iMat[i][j] = 0
			jMat[i][j] = 0

	
	iVec = <double *> malloc(3 * sizeof(double))
	jVec = <double *> malloc(3 * sizeof(double))
	for i in range (3):
		iVec[i] = 0
		jVec[i] = 0

	oiVec = <double *> malloc(3 * sizeof(double))
	ojVec = <double *> malloc(3 * sizeof(double))
	oijVec = <double *> malloc(3 * sizeof(double))
	for i in range (3):
		oiVec[i] = 0
		ojVec[i] = 0
		oijVec[i] =0 

	fVec = <double *> malloc(3 * sizeof(double))
	for i in range (3):
		fVec[i] = 0


	
	# Starting read coordinates
	frame = 0
	nFrame_used = 0
	for coordMat in coordtMat:

		if frame % 100 == 0:
			print frame

		#if frame == 10000:
		#	break

		if mVec[frame] != nMol:
			frame += 1
			continue

		frame += 1
		nFrame_used += 1

		
		# Calculate intermolecular force
		for i in range (nMol):
			for j in range (i+1, nMol):
				if i==j:
					continue

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

				cal_force(fVec, iMat, jMat, cut, apm)

				fMat[i,j,0] = fVec[0]
				fMat[i,j,1] = fVec[1]
				fMat[i,j,2] = fVec[2]

		# calculate com matrix using OW position
		comx = 0; comy = 0; comz = 0;
		for i in range (nMol):
			comx += 16*coordMat[apm*i,0] + coordMat[apm*i+1,0] + coordMat[apm*i+2,0]
			comy += 16*coordMat[apm*i,1] + coordMat[apm*i+1,1] + coordMat[apm*i+2,1]
			comz += 16*coordMat[apm*i,2] + coordMat[apm*i+1,2] + coordMat[apm*i+2,2]
			#comx += coordMat[apm*i,0]
			#comy += coordMat[apm*i,1]
			#comz += coordMat[apm*i,2]
		comx /= float(nMol)*18
		comy /= float(nMol)*18
		comz /= float(nMol)*18
		for i in range (apm*nMol):
			coordMat[i,0] -= comx
			coordMat[i,1] -= comy
			coordMat[i,2] -= comz



		# Calculate Pressure
		for i in range (nMol):
			oiVec[0] = (16*coordMat[apm*i,0] + coordMat[apm*i+1,0] + coordMat[apm*i+2,0])/18.
			oiVec[1] = (16*coordMat[apm*i,1] + coordMat[apm*i+1,1] + coordMat[apm*i+2,1])/18.
			oiVec[2] = (16*coordMat[apm*i,2] + coordMat[apm*i+1,2] + coordMat[apm*i+2,2])/18.

			ri2 = oiVec[0]*oiVec[0] + oiVec[1]*oiVec[1] + oiVec[2]*oiVec[2]

			for j in range (i+1, nMol):
				if i==j:
					continue 
				ojVec[0] = (16*coordMat[apm*j,0] + coordMat[apm*j+1,0] + coordMat[apm*j+2,0])/18.
				ojVec[1] = (16*coordMat[apm*j,1] + coordMat[apm*j+1,1] + coordMat[apm*j+2,1])/18.
				ojVec[2] = (16*coordMat[apm*j,2] + coordMat[apm*j+1,2] + coordMat[apm*j+2,2])/18.


				'''
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
				'''

				rj2 = ojVec[0]*ojVec[0] + ojVec[1]*ojVec[1] + ojVec[2]*ojVec[2]

				oijVec[0] = ojVec[0] - oiVec[0]
				oijVec[1] = ojVec[1] - oiVec[1]
				oijVec[2] = ojVec[2] - oiVec[2]

				rij2 = oijVec[0]*oijVec[0] + oijVec[1]*oijVec[1] + oijVec[2]*oijVec[2]
				ririj = oiVec[0]*oijVec[0] + oiVec[1]*oijVec[1] + oiVec[2]*oijVec[2]

				rij = sqrt(rij2)

				l02 = ri2 - (ririj/rij)**2
				l0 = sqrt(l02)

				fVec[0] = fMat[i,j,0]
				fVec[1] = fMat[i,j,1]
				fVec[2] = fMat[i,j,2]
				f = (fVec[0]*oijVec[0] + fVec[1]*oijVec[1] + fVec[2]*oijVec[2])/rij 

				'''
				print 'oiVec'
				print oiVec[0], oiVec[1], oiVec[2]
				print 'ojVec'
				print ojVec[0], ojVec[2], ojVec[2]
				print 'rij'
				print rij
				print 'fVec'
				print fVec[0], fVec[1], fVec[2]
				print 'f'
				print f
				exit(1)
				'''

				if l02 < 0:
					l0 = 0
				else:
					kMin = int(l0/dr)
					if kMin > len(rVec)-1:
						continue

					if rj2 > ri2:
						kMax = int(sqrt(rj2)/dr) + 1
					else:
						kMax = int(sqrt(ri2)/dr) + 1

					if kMax > len(rVec)-1:
						kMax = len(rVec)

					for k in range (kMin, kMax):
						r = rVec[k]

						Dout = ririj**2 - rij2*(ri2 - r*r)

						if Dout < FLT_EPSILON:
							continue
						
						loutp = (-ririj + sqrt(Dout))/rij2
						loutn = (-ririj - sqrt(Dout))/rij2

						pN = 0
						if loutn>1-FLT_EPSILON or loutp<FLT_EPSILON:
							pN = 0
						else:
							if loutn<-FLT_EPSILON:
								pN += 0
							else:
								pN += fabs(ririj + loutn*rij2)/(r*rij)
							
							if loutp>1+FLT_EPSILON:
								pN += 0
							else:
								pN += fabs(ririj + loutp*rij2)/(r*rij)

						pN *= f
						pnVec[k] += pN



	# prepare pK
	for i in range (len(rVec)):
		pkVec[i] = pref*kBT*dVec[i]
	
	for i in range (len(rVec)):
		pnVec[i] *= pref/nFrame_used/(4*pi*rVec[i]*rVec[i])
		ptVec[i] *= pref/nFrame_used/(4*pi*rVec[i]*rVec[i])
	
		
	print 'nFrame_used = {}'.format(nFrame_used)

	# prepare mechanical equilibrium
	pt2Vec = np.zeros(len(pkVec) )

	for i in range (len(pkVec)-1):
		pt2Vec[i] = (pkVec[i] + pnVec[i]) + 0.5*rVec[i]*(pkVec[i+1] + pnVec[i+1] - pkVec[i] - pnVec[i])/(rVec[1]-rVec[0])

	# coordt loop is over
	oMat = []
	oMat.append(rVec)
	oMat.append(pkVec)
	oMat.append(ptVec)
	oMat.append(pnVec)
	#oMat.append(pkVec+ptVec+pnVec)
	oMat.append(pt2Vec)
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


		


							






