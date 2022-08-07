import cv2
import numpy as np 
import random
import glob
import math
import time
import matplotlib.pyplot as plt
import argparse

def FunMat(src_pts, dst_pts) :
	src_pts, T1 = normalizePoints(src_pts)
	dst_pts, T2 = normalizePoints(dst_pts)

	A = np.zeros((src_pts.shape[0], 9)).astype(int)
	for i in range(src_pts.shape[0]) :
		s, d = src_pts[i], dst_pts[i]
		A[i, :] = np.array([s[0]*d[0], s[0]*d[1], s[0], s[1]*d[0], s[1]*d[1], s[1], d[0], d[1], 1])

	# solution of Af = 0 using SVD
	Ui, Si, Vi = np.linalg.svd(A)
	F = (Vi[-1,:])
	F = F.reshape((3,3))

	Ui_, Si_, Vi_ = np.linalg.svd(F)
	Si_ = np.diag(Si_)
	Si_[2,2] = 0

	# # constrain F
	# make rank 2 by zeroing out last singular value
	F = Ui_@(Si_@Vi_)

	# unnormalize Fmatrix
	F = T2.T@(F@T1)

	return F

def normalizePoints(pts):
	pts_mean = np.mean(pts, axis=0)
	x_bar = pts_mean[0]
	y_bar = pts_mean[1]

	# origin of the new coordinate system should be located at the centroid of the image points
	x_s, y_s = pts[:,0] - x_bar, pts[:, 1] - y_bar

	# scale by the scaling factor
	s = (2/np.mean(x_s**2 + y_s**2))**(0.5)

	# construct transformation matrix (translation+scaling)
	T_S = np.diag([s,s,1])
	T_T = np.array([[1, 0, -x_bar],[0, 1, -y_bar],[0, 0, 1]])
	Ta = np.dot(T_S, T_T)

	x = np.column_stack((pts, np.ones(pts.shape[0])))
	x_norm = (Ta@x.T).T
	return x_norm, Ta

def FunMat1(src_pts, dst_pts) :
	A = np.zeros((8, 9)).astype(int)
	for i in range(src_pts.shape[0]) :
		s, d = src_pts[i], dst_pts[i]
		A[i, :] = np.array([s[0]*d[0], s[0]*d[1], s[0], s[1]*d[0], s[1]*d[1], s[1], d[0], d[1], 1])

	# solution of Af = 0 using SVD
	Ui, Si, Vi = np.linalg.svd(A)
	F = (Vi[-1,:] / Vi[-1,-1])
	F = F.reshape((3,3))

	return F

def estimateFunMatrix(img1, img2) :
	gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

	# # Creating SIFT object
	sift = cv2.xfeatures2d.SIFT_create()

	kpts1, descs1 = sift.detectAndCompute(gray1,None)
	kpts2, descs2 = sift.detectAndCompute(gray2,None)

	# feature matching
	bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck = True)
	matches = bf.match(descs1, descs2)
	dmatches = sorted(matches, key = lambda x:x.distance)

	## extract the matched keypoints
	src_pts  = np.array([kpts1[m.queryIdx].pt for m in dmatches]).astype(int)
	dst_pts  = np.array([kpts2[m.trainIdx].pt for m in dmatches]).astype(int)

	# drawMatches(img1, img2, src_pts, dst_pts)

	# Getting the inliers using RANSAC
	num_iterations = 2000 # default is 5000
	rows = src_pts.shape[0]
	max_error = 0.001
	Fmatrix = np.zeros((3, 3))
	maxinliers = 0
	finalidx = []

	for i in range(num_iterations) :
		indexes = random.sample(range(0, rows), 8)
		src = src_pts[indexes]
		dst = dst_pts[indexes]
		FM = FunMat(src, dst)
		test = np.zeros((rows, 1))
		tempidx = []
		for j in range(rows) :
			error = np.abs(np.array([[src_pts[j][0]], [src_pts[j][1]], [1]]).T @ FM @ np.array([[dst_pts[j][0]], [dst_pts[j][1]], [1]]))
			if error < max_error :
				tempidx.append(j)

		if len(tempidx) > maxinliers:
			maxinliers = len(tempidx)
			finalidx = tempidx
			Fmatrix = FM
			srcFinal = src_pts[tempidx]
			dstFinal = dst_pts[tempidx]

	print('---Fmatrix---')
	Fnmatrix = FunMat(srcFinal, dstFinal)
	Fnmatrix = Fnmatrix/Fnmatrix[2,2]
	print(Fnmatrix)
	return Fnmatrix, dstFinal, srcFinal

def drawMatches(img1, img2, keypoints1, keypoints2) :
	new_img = np.concatenate((img1, img2), axis=1)
	numkeypoints = len(keypoints1)
	r = 4
	thickness = 1
	
	for i in range(keypoints1.shape[0]) :
		end1 = keypoints1[i]
		end2 = (keypoints2[i][0]+img1.shape[1], keypoints2[i][1])

		cv2.line(new_img, end1, end2, (0,255,255), thickness)
		cv2.circle(new_img, end1, r, (0,0,255), thickness)
		cv2.circle(new_img, end2, r, (0,255,0), thickness)

	resized = cv2.resize(new_img, (1440, 720), interpolation = cv2.INTER_AREA)
	cv2.imshow('SIFT Matches', resized)
	cv2.waitKey(0)

def getEssentMatrix(Fmatrix, flag) :
	if flag == 1 :
		# Pendulumn
		## R1 C1
		K1 = np.array([[1729.05, 0, -364.24], [0, 1729.05, 552.22], [0, 0, 1]])
		K2 = np.array([[1729.05, 0, -364.24], [0, 1729.05, 552.22], [0, 0, 1]])
		b = 537.75
		f = K1[0, 0]
		vmin = 25
		vmax = 150 
		ndisp = 180

	if flag == 2 :
		# Octagon
		K1 = np.array([[1742.11, 0, 804.90], [0, 1742.11, 541.22], [0, 0, 1]])
		K2 = np.array([[1742.11, 0, 804.90], [0, 1742.11, 541.22], [0, 0, 1]])
		b = 221.76
		f = K1[0, 0]
		vmin = 29 
		vmax = 61
		ndisp = 100

	if flag == 3 :
		# curule
		## R4 C4
		K1 = np.array([[1758.23, 0, 977.42], [0, 1758.23, 552.15], [0, 0, 1]])
		K2 = np.array([[1758.23, 0, 977.42], [0, 1758.23, 552.15], [0, 0, 1]])
		b = 88.39
		f = K1[0, 0]
		vmin = 55
		vmax = 195
		ndisp = 220

	Ematrix = K1.T @ Fmatrix @ K2

	# singular values of E are not necessarily (1,1,0)
	U, D, V = np.linalg.svd(Ematrix) 
	A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
	Ematrix = U@A@V
	print()
	print('---Ematrix---')
	print(Ematrix)

	return Ematrix, K1, K2, b, f, vmin, vmax, ndisp

def getCameraPose(Ematrix) :
	U, D, V = np.linalg.svd(Ematrix)
	W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

	C1, C2, C3, C4 = U[:,2], -U[:,2], U[:,2], -U[:,2]
	R1, R2, R3, R4 = U@W@V, U@W@V, U@W.T@V, U@W.T@V
	if np.linalg.det(R1) < 0 : 
		C1 = np.negative(C1)
		R1 = np.negative(R1)

	if np.linalg.det(R2) < 0 : 
		C2 = np.negative(C2)
		R2 = np.negative(R2)

	if np.linalg.det(R3) < 0  : 
		C3 = np.negative(C3)
		R3 = np.negative(R3)

	if np.linalg.det(R4) < 0  : 
		C4 = np.negative(C4)
		R4 = np.negative(R4)

	return C1, C2, C3, C4, R1, R2, R3, R4

def chiralityCheck(C, R, K1, K2, dstFinal, srcFinal) :
	maxcount = 0
	Cf, Rf = C[-1], R[-1]
	for i in range(4) :
		Ci = C[i]
		Ri = R[i] 
		Ci = Ci.reshape((3,1))

		#RT matrix for C1 is identity.
		R1 = np.identity(3)
		T1 = np.zeros((3,1))
		I = np.identity(3)

		P1 = np.dot(K1, np.dot(R1, np.hstack((I, -T1.reshape(3,1)))))
		P2 = np.dot(K2, np.dot(Ri, np.hstack((I, -Ti.reshape(3,1)))))

		# RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
		# P1 = K1 @ RT1 #projection matrix for C1
		 
		## RT matrix for C2 is the R and T obtained from stereo calibration.
		# RT2 = np.concatenate([Ri, Ci], axis = -1)
		# P2 = K2 @ RT2 #projection matrix for C2
		count = 0
		for j in range(srcFinal.shape[0]) :
			src = srcFinal[i]
			dst = dstFinal[i]
			src3d = cv2.triangulatePoints(P1, P2, src.T, dst.T)
			src3d = src3d/src3d[3]
				
			r3 = Ri[2, :]
			print((src3d[:3] - Ci))
			print(r3)
			if r3@(src3d[:3] - Ci).flatten() > 0 :
				count += 1 

		if count > maxcount :
			Cf = Ci
			Rf = Ri 
			num = i 
			maxcount = count 

	return Cf, Rf, num

def stereo(path, scaling, windowsize, flag=1) :
	image_list = []
	for filename in glob.glob(f'{path}/*.png'): 
		im = cv2.imread(filename)
		image_list.append(im)

	img1 = image_list[0]
	img2 = image_list[1]
	Fmatrix, dstFinal, srcFinal = estimateFunMatrix(img1, img2)

	drawMatches(img1, img2, srcFinal, dstFinal)
	Ematrix, K1, K2, b, f, vmin, vmax, ndisp = getEssentMatrix(Fmatrix, flag)
	C1, C2, C3, C4, R1, R2, R3, R4 = getCameraPose(Ematrix)
	C = [C1, C2, C3, C4]
	R = [R1, R2, R3, R4]
	print('C and R')
	print(C1)
	print(C2)
	print(C3)
	print(C4)

	print()
	print(R1)
	print(R2)
	print(R3)
	print(R4)

	## Cheirality Check for choosing R and C
	# Cf, Rf, num = chiralityCheck(C, R, K1, K2, dstFinal, srcFinal)
	# print(f'{num+1} is the best')

	## Get epipole line
	els, eld = getEpipoleLine(srcFinal, dstFinal, Fmatrix)
	# drawEpipoleLine(els, eld, img1, img2, srcFinal, dstFinal)

	## Get epipoles 
	# eps, epd = getEpipoles(els, eld) 

	## Rectification
	# H1, H2 = rectify(srcFinal, dstFinal, Fmatrix, eps, epd, img1, img2) 
	ret, H1, H2 = cv2.stereoRectifyUncalibrated(srcFinal, dstFinal, Fmatrix, (img1.shape[1], img1.shape[0]))
	R1 = H1
	R2 = H2
	print('---Homography---')
	R1 = R1 / R1[2,2]
	R2 = R2 / R2[2,2]
	print(R1)
	print(R2)

	img1c = img1.copy()
	img2c = img2.copy()
	drawEpipoleLine(els, eld, img1c, img2c, srcFinal, dstFinal)

	trans_img1 = cv2.warpPerspective(img1, R1, (img1.shape[1], img1.shape[0]))
	trans_img2 = cv2.warpPerspective(img2, R2, (img1.shape[1], img1.shape[0]))

	trans_img1c = cv2.warpPerspective(img1c, R1, (img1.shape[1], img1.shape[0]))
	trans_img2c = cv2.warpPerspective(img2c, R2, (img1.shape[1], img1.shape[0]))
	nFmatrix = np.linalg.inv(H2.T) @ Fmatrix @ np.linalg.inv(H1)

	# drawEpipoleLine(els, eld, trans_img1, trans_img2, nsrcFinal, ndstFinal)
	scale_percent = scaling
	scalingfactor = scale_percent / 100
	width = int(img1.shape[1] * scalingfactor)
	height = int(img1.shape[0] * scalingfactor)
	ims = (width, height)
	ndisp = int(ndisp * scalingfactor)

	res1 = cv2.resize(trans_img1, ims, interpolation = cv2.INTER_AREA)
	res2 = cv2.resize(trans_img2, ims, interpolation = cv2.INTER_AREA)
	new_img = np.concatenate((res1, res2), axis=1)

	## Visulization
	res1c = cv2.resize(trans_img1c, ims, interpolation = cv2.INTER_AREA)
	res2c = cv2.resize(trans_img2c, ims, interpolation = cv2.INTER_AREA)
	new_imgc = np.concatenate((res1c, res2c), axis=1)

	cv2.imshow('Warp1', res1)
	cv2.imshow('Warp2', res2)
	cv2.imshow('Rectified', new_img)
	cv2.waitKey(0)

	cv2.imshow('Warp1 EpipoleLines', res1c)
	cv2.imshow('Warp2 EpipoleLines', res2c)
	cv2.imshow('Rectified EpipoleLines', new_imgc)
	cv2.waitKey(0)

	resized1 = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)
	resized2 = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)

	time1 = time.time()
	dimg, dpimg, dcimg, dpcimg = getDisparityMap(resized1, resized2, b, f, vmin, vmax, ndisp, windowsize)

	cv2.imwrite(f'DisparityHeat{flag}.png', dcimg)
	cv2.imwrite(f'DepthHeat{flag}.png', dpcimg)

	cv2.imwrite(f'DisparityGray{flag}.png', dimg)
	cv2.imwrite(f'DepthGray{flag}.png', dpimg)
	time2 = time.time()
	print(f'Time taken : {time2-time1}')
	
## Disparity Map 
def getDisparityMap(img1, img2, b, f, vmin, vmax, ndisp, winsize) :
	# left on right
	img1c = img1.copy()
	h, w = img1c.shape[0], img1c.shape[1]  
	windowSize = winsize
	halfWindow = math.floor((windowSize)/2)
	disparity = np.zeros((h, w), np.uint8)
	stride = 1

	print('Search : ', ndisp)
	for r in range(halfWindow, h-halfWindow, stride) :
		for c in range(halfWindow, w-halfWindow, stride) :
			pimg1 = img1[r-halfWindow : r+halfWindow, c-halfWindow : c+halfWindow]
			min_ssd = 15000 
			disp = 0
			searchDistance = ndisp

			## search on same parallel line in second image
			## search before and after some distance from that point
			for distance in range(-searchDistance, searchDistance): 
				c_dash = c + distance
				if (c_dash < w - halfWindow) and (c_dash > halfWindow):
					pimg2 = img2[r - halfWindow: r + halfWindow, c_dash - halfWindow : c_dash + halfWindow]
					ssd = np.sum((pimg1-pimg2)**2)
					if ssd < min_ssd:
						min_ssd = ssd
						disp = np.abs(distance)
			disparity[r, c] = disp

		print(r+1)

	## Rescaling the disparity map between 0-255
	disparity[disparity < 1] = 1
	disparity[disparity > np.mean(disparity)] = np.mean(disparity)
	disparity = imgnormalize(disparity)
	disparity = np.uint8(disparity * (255 / np.max(disparity)))

	## Depth
	depth = 255-disparity 
	# depth = (b * f) / (disparity+1e-10)
	# depth[depth > b*f] = b*f
	# depth = np.uint8(depth * 255 / np.max(depth))

	disparityim = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)
	depthim = cv2.applyColorMap(depth, cv2.COLORMAP_JET)

	return disparity, depth, disparityim, depthim

def imgnormalize(image, maxm = 255, minm = 0):
	image = ((image-image.min())*((maxm-minm)/(image.max()-image.min())))+minm
	return image

def getEpipoleLine(srcFinal, dstFinal, F) :
	els = np.zeros((srcFinal.shape[0], 3))
	eld = np.zeros((dstFinal.shape[0], 3))

	for i in range(srcFinal.shape[0]) :
		els[i] = F@np.array([dstFinal[i][0], dstFinal[i][1], 1])

	for i in range(dstFinal.shape[0]) :
		eld[i] = F@np.array([srcFinal[i][0], srcFinal[i][1], 1])

	return eld, els

def getEpipoles(els, eld) :
	Usi, Ssi, Vsi = np.linalg.svd(els)
	eps = Vsi[-1,:]

	Udi, Sdi, Vdi = np.linalg.svd(eld)
	epd = Vdi[-1,:]
	return eps, epd

def rectify(srcFinal, dstFinal, Fmatrix, e1, e2, img1, img2) :
	width = img2.shape[1]
	height = img2.shape[0]

	points1 = np.column_stack((srcFinal, np.ones(srcFinal.shape[0])))
	points2 = np.column_stack((dstFinal, np.ones(dstFinal.shape[0])))

	T = np.identity(3)
	T[0][2] = -width/2
	T[1][2] = -height/2

	e = T.dot(e2)
	e1_prime = e[0]
	e2_prime = e[1]

	if e1_prime >= 0:
		alpha = 1.0
	else:
		alpha = -1.0

	R = np.identity(3) 
	R[0][0] = alpha*e1_prime/(e1_prime**2 + e2_prime**2)**0.5
	R[1][0] = -alpha*e2_prime/(e1_prime**2 + e2_prime**2)**0.5
	R[0][2] = alpha*e2_prime/(e1_prime**2 + e2_prime**2)**0.5
	R[1][1] = alpha*e1_prime/(e1_prime**2 + e2_prime**2)**0.5

	G = np.identity(3)
	f = R.dot(e)[0]
	G = np.identity(3)
	G[2][0] = - 1.0 / f

	H2 = np.linalg.inv(T)@G@R@T

	ex = np.zeros((3, 3))
	ex[0][1] = -e2[2]
	ex[0][2] = e2[1]
	ex[1][0] = e2[2]
	ex[1][2] = -e2[0]
	ex[2][0] = -e2[1]
	ex[2][1] = e2[0]

	v = np.array([1, 1, 1])
	M = ex@Fmatrix + np.outer(e2, v)
	print(M.shape)
	print(points1.shape)
	print(H2.shape)

	points1_hat = H2.dot(M.dot(points1.T)).T
	points2_hat = H2.dot(points2.T).T

	W = points1_hat / points1_hat[:, 2].reshape(-1, 1)
	b = (points2_hat / points2_hat[:, 2].reshape(-1, 1))[:, 0]

	# least square problem
	a1, a2, a3 = np.linalg.lstsq(W, b)[0]
	HA = np.identity(3)
	HA[0] = np.array([a1, a2, a3])
	H1 = HA@H2@M

	print('---Homography---')
	print(H1)
	print(H2)
	return H1, H2

def drawEpipoleLine(els, eld, img1, img2, srcFinal, dstFinal) :
	# plotting epipole line 
	new_img = np.concatenate((img1, img2), axis=1)
	r = 4
	thickness = 2
	numpts = 50
	for i in range(numpts) :
		elsi = els[i]
		a, b, c = elsi[0], elsi[1], elsi[2]
		y1coord = int((a*0+c)/(-b))
		y2coord = int((a*img1.shape[1]+c)/(-b))
		cv2.line(img2, (0, y1coord), (img2.shape[1], y2coord), (0, 0, 255), 2)
		cv2.circle(img2, (int(dstFinal[i][0]), int(dstFinal[i][1])), r, (0,0,255), thickness)

	for i in range(numpts) :
		eldi = eld[i]
		a, b, c = eldi[0], eldi[1], eldi[2]
		y1coord = int((a*0+c)/(-b))
		y2coord = int((a*img2.shape[1]+c)/(-b))
		cv2.line(img1, (0, y1coord), (img1.shape[1], y2coord), (0, 255, 0), 2)
		cv2.circle(img1, (int(srcFinal[i][0]), int(srcFinal[i][1])), r, (0, 255, 0), thickness)

	new_img = np.hstack((img1, img2))
	resized = cv2.resize(new_img, (1440, 720), interpolation = cv2.INTER_AREA)
	# resized = cv2.resize(new_img, (900, 600), interpolation = cv2.INTER_AREA)
	cv2.imshow('NewImage', resized)
	cv2.waitKey(0)

if __name__ == "__main__" :

	Parser = argparse.ArgumentParser()
	Parser.add_argument('--DirPath', default="./octagon", help='base path where data files exist')
	Parser.add_argument('--Flag', default="2", help='flag to get the info about the folder')
	Parser.add_argument('--Scaling', default="33", help='scaling percent where data files exist')
	Parser.add_argument('--WindowSize', default="11", help='window size for disparity')
	Args = Parser.parse_args("")
	DirPath = Args.DirPath
	Flag = Args.Flag
	Scaling = Args.Scaling
	WindowSize = Args.WindowSize
	path = DirPath
	flag = Flag
	scaling = Scaling
	stereo(path, int(scaling), int(WindowSize), int(flag))
