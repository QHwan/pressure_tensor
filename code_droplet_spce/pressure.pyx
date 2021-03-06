import cython
cimport cython
import numpy as np
cimport numpy as np
import sys
import math
from libc.math cimport sqrt, log, pi, atan
from libc.stdlib cimport malloc, free
import time
from mdtraj.formats import XTCTrajectoryFile
from mdtraj.formats import TRRTrajectoryFile



from func cimport integral_range, cal_force, cal_pn, cal_pt, vec_dot, vec_minus


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def main():

	# This line we declare the variables
	cdef int i, j, k, l, m
	cdef int frame, nFrame, nFrame_used, nMol, apm
	cdef int lenCoordVec
	cdef int kMin
	cdef int rowLength
	cdef double kBT, T
	cdef double doh, dom
	cdef double f, t
	cdef double comx, comy, comz
	cdef double L, L2
	cdef double rMin, rMax, dr
	cdef double x, y, z, x0, y0, z0
	cdef double ri, rj, rij, ri2, rj2, rij2, ririj
	cdef double rxyi, rxyj, rxyij, rxyi2, rxyj2, rxyij2, rxyirxyij
	cdef double Din, Dout, sDin, sDout, Rin, Rout, Rin2, Rout2, Da, Db
	cdef double La, Lb
	cdef double linn, linp, loutn, loutp, ll, l0, l02
	cdef double lxy0, lxy02, lxy2, lxy
	cdef double la, lb, lap, lbp
	cdef double pN, pT, pK
	cdef double v
	cdef double pref
	cdef double cut

	cdef np.ndarray[float, ndim=3, mode='c'] coordtMat
	cdef np.ndarray[float, ndim=2, mode='c'] coordMat
	cdef np.ndarray[float, ndim=1] tVec
	cdef np.ndarray[double, ndim=1] rVec
	cdef np.ndarray[float, ndim=3] boxMat
	cdef np.ndarray[double, ndim=1] pkVec
	cdef np.ndarray[double, ndim=1, mode='c'] buf_pnVec
	cdef np.ndarray[double, ndim=1, mode='c'] buf_ptVec
	cdef np.ndarray[double, ndim=1] vVec

	cdef np.ndarray[double, ndim=3, mode='c'] fMat

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

	cdef double *Rin2Vec
	cdef double *Rout2Vec


	

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

	print 'L = {}'.format(L)

	pref = 1.6602	# kJ/mol/nm3 -> MPa
	T = 300.
	kBT = 2.479*T/298

	cut = 4.0

	# Setting PMat
	rowLength = len(rVec)
	dr = rVec[1]-rVec[0]
	rMin = dr*0.5
	rMax = rVec[rowLength-1]


	Rin2Vec = <double *> malloc (rowLength*sizeof(double))
	Rout2Vec = <double *> malloc (rowLength*sizeof(double))

	for i in range (rowLength):
		Rin2Vec[i] = (rVec[i] - 0.5*dr)**2
		Rout2Vec[i] = (rVec[i] + 0.5*dr)**2

	pkVec = np.zeros( rowLength)
	pnMat = []
	ptMat = []


	vVec = np.zeros(rowLength)
	for i in range (rowLength):
		vVec[i] = 4*math.pi/3* (pow(rVec[i] + dr*0.5,3) - pow(rVec[i] - dr*0.5,3))


	fMat = np.zeros( (nMol,nMol, 3))

	if rowLength != len(dVec):
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

		buf_pnVec = np.zeros(rowLength)
		buf_ptVec = np.zeros(rowLength)

		if frame % 100 == 0:
			print frame

		#if frame == 1:
		#	break

		if mVec[frame] != nMol:
			frame += 1
			continue

		frame += 1
		nFrame_used += 1


		# calculate com matrix using OW position
		comx = 0; comy = 0; comz = 0;
		for i in range (nMol):
			comx += 16*coordMat[apm*i,0] + coordMat[apm*i+1,0] + coordMat[apm*i+2,0]
			comy += 16*coordMat[apm*i,1] + coordMat[apm*i+1,1] + coordMat[apm*i+2,1]
			comz += 16*coordMat[apm*i,2] + coordMat[apm*i+1,2] + coordMat[apm*i+2,2]
		comx /= 18.*nMol
		comy /= 18.*nMol
		comz /= 18.*nMol
		for i in range (apm*nMol):
			coordMat[i,0] -= comx
			coordMat[i,1] -= comy
			coordMat[i,2] -= comz

		# Calculate intermolecular force
		for i in range (nMol):
			for j in range (i+1, nMol):

				for k in range (apm):
					iMat[k][0] = coordMat[apm*i+k,0]
					iMat[k][1] = coordMat[apm*i+k,1]
					iMat[k][2] = coordMat[apm*i+k,2]
					jMat[k][0] = coordMat[apm*j+k,0]
					jMat[k][1] = coordMat[apm*j+k,1]
					jMat[k][2] = coordMat[apm*j+k,2]

				
				fVec[0] = 0
				fVec[1] = 0
				fVec[2] = 0

				cal_force(fVec, iMat, jMat, cut, apm)

				fMat[i,j,0] = fVec[0]
				fMat[i,j,1] = fVec[1]
				fMat[i,j,2] = fVec[2]



		# Calculate Pressure
		for i in range (nMol):
			oiVec[0] = (16*coordMat[apm*i,0] + coordMat[apm*i+1,0] + coordMat[apm*i+2,0])/18. 
			oiVec[1] = (16*coordMat[apm*i,1] + coordMat[apm*i+1,1] + coordMat[apm*i+2,1])/18.
			oiVec[2] = (16*coordMat[apm*i,2] + coordMat[apm*i+1,2] + coordMat[apm*i+2,2])/18.

			ri2 = oiVec[0]*oiVec[0] + oiVec[1]*oiVec[1] + oiVec[2]*oiVec[2]

			for j in range (i+1, nMol):

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

				l02 = ri2 - (ririj*ririj/rij2)
				if l02 < 0:
					continue
				l0 = sqrt(l02)

				fVec[0] = fMat[i,j,0]
				fVec[1] = fMat[i,j,1]
				fVec[2] = fMat[i,j,2]

				kMin = int(l0/dr)
				if kMin > rowLength-1:
					continue

				if rj2 > ri2:
					kMax = int(sqrt(rj2)/dr) + 1
				else:
					kMax = int(sqrt(ri2)/dr) + 1

				if kMax > rowLength-1:
					kMax = rowLength

				for k in range (kMin, kMax, 1):
					r = rVec[k]

					# Check Din and Dout
					Din = (ririj*ririj - rij2*(ri2 - Rin2Vec[k]))
					Dout = (ririj*ririj - rij2*(ri2 - Rout2Vec[k]))

					if Dout <= 0:
						continue
					
					loutp = (-ririj + sqrt(Dout))/rij2
					loutn = (-ririj - sqrt(Dout))/rij2

					if loutn>1 or loutp<0:
						continue

					if Din > 0:
						linp = (-ririj + sqrt(Din))/rij2
						linn = (-ririj - sqrt(Din))/rij2
					else:
						linp = -10
						linn = -10

					if linn<0 and linp>1:
						continue

					# choose integral range
					la, lb, lap, lbp = integral_range(Din, loutn, linn, linp, loutp)

					buf_pnVec[k] += cal_pn(fVec, oiVec, oijVec, la, lb, ri2, rij2, ririj)
					buf_ptVec[k] += cal_pt(fVec, oiVec, oijVec, la, lb)

					if lbp > lap:
						buf_pnVec[k] += cal_pn(fVec, oiVec, oijVec, lap, lbp, ri2, rij2, ririj)
						buf_ptVec[k] += cal_pt(fVec, oiVec, oijVec, lap, lbp)

						if lap<0 or lbp<0:
							print Din, la, lb, lap, lbp

		for i in range (rowLength):
			buf_pnVec[i] *= pref/vVec[i]
			buf_ptVec[i] *= pref/vVec[i]

		pnMat.append(buf_pnVec)
		ptMat.append(buf_ptVec)

	# prepare pK
	for i in range (rowLength):
		pkVec[i] = pref*kBT*dVec[i]
	
	print 'nFrame_used = {}'.format(nFrame_used)

	pnMat = np.asarray(pnMat)
	pnMat = np.transpose(pnMat)

	ptMat = np.asarray(ptMat)
	ptMat = np.transpose(ptMat)

	pnVec = np.zeros(rowLength)
	pnsVec = np.zeros(rowLength)

	ptVec = np.zeros(rowLength)
	ptsVec = np.zeros(rowLength)

	for i in range (rowLength):
		pnVec[i] += np.mean(pnMat[i])	
		pnsVec[i] += np.std(pnMat[i])/sqrt(float(nFrame_used))

		ptVec[i] += np.mean(ptMat[i])
		ptsVec[i] += np.std(ptMat[i])/sqrt(float(nFrame_used))


	# prepare mechanical equilibrium
	pt2Vec = np.zeros(len(pkVec) )

	for i in range (len(pkVec)-1):
		pt2Vec[i] = (pkVec[i] + buf_pnVec[i]) + 0.5*rVec[i]*(pkVec[i+1] + buf_pnVec[i+1] - pkVec[i] - buf_pnVec[i])/(rVec[1]-rVec[0])

	# coordt loop is over
	oMat = []
	oMat.append(rVec)
	oMat.append(pkVec)
	oMat.append(pnVec)
	oMat.append(pnsVec)
	oMat.append(ptVec)
	oMat.append(ptsVec)
	#oMat.append(pkVec+buf_ptVec+buf_pnVec)
	#oMat.append(pt2Vec)
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

	free(Rin2Vec)
	free(Rout2Vec)

	free(fVec)



		


							






