write_images = True
nb_instances = 0
import math
import cv2

#ifndef sign
#define sign(a) ((a>=0)?1:(-1))
#endif
#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

class marker_detection:
	def __init__(self, camera: int, frame: int, marker: int, search_area: int, threshold_offset: int = 8, m_input_size: int):
		self.search_area = search_area + 0.5 if search_area >= 9.5 else 10
		self.m_input_size = m_input_size if m_input_size > 0 else 5
		self.marker_x = {pre:None , post: None}
		self.marker_y = {pre:None , post: None}
		print(self.search_area)
		print(self.m_input_size)
		print(self.marker_x)
		print(self.marker_y)

	def detect_marker(self):
		pt = detection_point()
		m_x = pt.x
		m_y = pt.y
		a = None



MarkerDetection.MarkerDetection(int camera, trial, frame, marker, searcharea, refinementAfterTracking) : QObject()



def detectMarker_thread(self):

	pt = detectionPoint(Project.getInstance().getTrials()[m_trial].getVideoStreams()[m_camera].getImage(), m_method, cv.Point2d(m_x, m_y), m_searchArea, m_input_size, m_thresholdOffset, &m_size)



#
#
# def detectMarker(self):
# 	m_FutureWatcher = QFutureWatcher<void>()
# 	m_FutureWatcher.finished.connect(self.detectMarker_threadFinished)
#
# 	QFuture<void> future = QtConcurrent.run(self, &MarkerDetection.detectMarker_thread)
# 	m_FutureWatcher.setFuture(future)


def detectionPoint(self, image, method, center, searchArea, masksize, threshold, size, std.vector <cv.Mat> * images, drawCrosshairs):
	if images != NULL) images.clear(:

	cv.Point2d point_out(center.x, center.y)
	double tmp_size

	off_x = (int)(center.x - searchArea + 0.5)
	off_y = (int)(center.y - searchArea + 0.5)

	#preprocess image
	cv.Mat subimage
	image.getSubImage(subimage, searchArea, off_x, off_y)

#ifdef WRITEIMAGES
	cv.Mat orig2
	cv.cvtColor(subimage, orig2, CV_GRAY2RGB)
	cv.imwrite("1_Det_original.png", orig2)
#endif

	if method == 0 or method == 2 or method == 5:
		if (method == 2) subimage = cv.Scalar.all(255) - subimage

		if images != NULL:
			cv.Mat tmp
			subimage.copyTo(tmp)
			images.push_back(tmp)


		#Convert To float
		cv.Mat img_float
		subimage.convertTo(img_float, CV_32FC1)
#ifdef WRITEIMAGES
		cv.Mat orig
		cv.cvtColor(subimage, orig, CV_GRAY2RGB)
#endif

		#Create Blurred image
		radius = (int)(1.5 * masksize + 0.5)
		sigma = radius * sqrt(2 * log(255)) - 1
		cv.Mat blurred
		cv.GaussianBlur(img_float, blurred, cv.Size(2 * radius + 1, * radius + 1), sigma)

#ifdef WRITEIMAGES
		cv.imwrite("2_Det_blur.png", blurred)
#endif
		if images != NULL:
			cv.Mat tmp
			blurred.convertTo(tmp,CV_8U)
			images.push_back(tmp)


		#Substract Background
		diff = img_float - blurred
		cv.normalize(diff, diff, 0, 255, cv.NORM_MINMAX, -1, cv.Mat())
		diff.convertTo(subimage, CV_8UC1)

#ifdef WRITEIMAGES
		cv.imwrite("3_Det_diff.png", diff)
#endif
		if images != NULL:
			cv.Mat tmp
			subimage.copyTo(tmp)
			images.push_back(tmp)


		#Median
		cv.medianBlur(subimage, subimage, 3)

#ifdef WRITEIMAGES
		cv.imwrite("4_Det_med.png", subimage)
#endif
		if images != NULL:
			cv.Mat tmp
			subimage.copyTo(tmp)
			images.push_back(tmp)


		#Thresholding
		double minVal
		double maxVal
		minMaxLoc(subimage, &minVal, &maxVal)
		thres = 0.5 * minVal + 0.5 * subimage.at<uchar>(searchArea, searchArea) + threshold * 0.01 * 255
		cv.threshold(subimage, subimage, thres, 255, cv.THRESH_BINARY_INV)
		#cv.adaptiveThreshold(image, image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 15, 10)
#ifdef WRITEIMAGES
		#fprintf(stderr, "Thres %lf Selected %d\n", thres, image.at<uchar>(searchArea, searchArea))
		cv.imwrite("5_Det_thresh.png", subimage)
#endif
		if images != NULL:
			cv.Mat tmp
			subimage.copyTo(tmp)
			images.push_back(tmp)


		cv.GaussianBlur(subimage, subimage, cv.Size(3, 3), 1.3)
#ifdef WRITEIMAGES
		#fprintf(stderr, "Thres %lf Selected %d\n", thres, image.at<uchar>(searchArea, searchArea))
		cv.imwrite("6_Det_threshBlur.png", subimage)
#endif
		if images != NULL:
			cv.Mat tmp
			subimage.copyTo(tmp)
			images.push_back(tmp)


		#Find contours
		cv.vector<cv.vector<cv.Point> > contours
		cv.vector<cv.Vec4i> hierarchy
		cv.findContours(subimage, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv.Point(off_x, off_y))
		dist = 1000
		bestIdx = -1

		#Find closest contour
		for (unsigned i = 0; i < contours.size(); i++)
			cv.Point2f detected_center

			if method == 5:
				mu = moments(contours[i], False)
				detected_center = cv.Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00)

			else:
				float circle_radius
				cv.minEnclosingCircle(contours[i], detected_center, circle_radius)


			distTmp = sqrt((center.x - detected_center.x) * (center.x - detected_center.x) + (center.y - detected_center.y) * (center.y - detected_center.y))
			if distTmp < dist:
				bestIdx = i
				dist = distTmp



		#set contour
		if bestIdx >= 0:
			cv.Point2f detected_center

			float circle_radius
			cv.minEnclosingCircle(contours[bestIdx], detected_center, circle_radius)

			if method == 5:
				mu = moments(contours[bestIdx], False)
				detected_center = cv.Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00)


			point_out.x = detected_center.x
			point_out.y = detected_center.y
			tmp_size = circle_radius


#ifdef WRITEIMAGES
			cent = circle_center - cv.Point2f(off_x, off_y)
			cv.circle(orig, cent, circle_radius, cv.Scalar(0, 0, 255, 50))
			std.cerr << circle_radius << std.endl
			std.cerr << cent.x << " " << cent.y << std.endl

			cv.imwrite("7_Detected.png", orig)
#endif
			if images != NULL:
				cv.Mat tmp
				image.getSubImage(subimage, searchArea, off_x, off_y)
				subimage.copyTo(tmp)
				images.push_back(tmp)

				if drawCrosshairs:					for (auto im : *images)						cent = detected_center - cv.Point2f(off_x, off_y)
						cv.line(im, cent - cv.Point2d(2, 0), cent + cv.Point2d(2, 0), cv.Scalar(127))
						cv.line(im, cent - cv.Point2d(0, 2), cent + cv.Point2d(0, 2), cv.Scalar(127))





#ifdef WRITEIMAGES
		else:
			fprintf(stderr, "Not found\n")


		#fprintf(stderr, "Stop Marker Detection : Camera %d Pos %lf %lf Size %lf\n", m_camera, x, y, size)
#endif
		#clean
		img_float.release()
		blurred.release()
		diff.release()
		hierarchy.clear()
		contours.clear()

	elif method == 3:
		imgGrey = IplImage(subimage)
		w = imgGrey.width
		h = imgGrey.height
		eig_image = cvCreateImage(cvSize(w, h), IPL_DEPTH_32F, 1)
		temp_image = cvCreateImage(cvSize(w, h), IPL_DEPTH_32F, 1)

		CvPoint2D32f corners[50] = {0
		corner_count = 50
		quality_level = 0.0001
		min_distance = 3
		eig_block_size = 7
		use_harris = True
		cvGoodFeaturesToTrack(imgGrey, eig_image, temp_image, corners, &corner_count, quality_level, min_distance,NULL, eig_block_size, use_harris)

		half_win_size = 7
		iteration = 100
		epislon = 0.001
		cvFindCornerSubPix(imgGrey, corners, corner_count, cvSize(half_win_size, half_win_size), cvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, iteration, epislon))

		dist_min = searchArea * searchArea
		double dist
		for (i = 0; i < corner_count; i ++)
			dist = cv.sqrt((corners[i].x - (searchArea + 1)) * (corners[i].x - (searchArea + 1)) + (corners[i].y - (searchArea + 1)) * (corners[i].y - (searchArea + 1)))
			if dist < dist_min:
				point_out.x = off_x + corners[i].x
				point_out.y = off_y + corners[i].y
				tmp_size = half_win_size * 2 + 1
				dist_min = dist


		cvReleaseImage(&eig_image)
		cvReleaseImage(&temp_image)

	elif method == 1 or method == 6:
		cv.SimpleBlobDetector.Params paramsBlob
		paramsBlob.thresholdStep = Settings.getInstance().getFloatSetting("BlobDetectorThresholdStep")
		paramsBlob.minThreshold = Settings.getInstance().getFloatSetting("BlobDetectorMinThreshold")
		paramsBlob.maxThreshold = Settings.getInstance().getFloatSetting("BlobDetectorMaxThreshold")
		paramsBlob.minRepeatability = Settings.getInstance().getIntSetting("BlobDetectorMinRepeatability")
		paramsBlob.minDistBetweenBlobs = Settings.getInstance().getFloatSetting("BlobDetectorMinDistBetweenBlobs")

		paramsBlob.filterByColor = Settings.getInstance().getBoolSetting("BlobDetectorFilterByColor")
		if method == 1:
			paramsBlob.blobColor = 255 - Settings.getInstance().getIntSetting("BlobDetectorBlobColor")

		elif method == 6:
			paramsBlob.blobColor = Settings.getInstance().getIntSetting("BlobDetectorBlobColor")

		paramsBlob.filterByArea = Settings.getInstance().getBoolSetting("BlobDetectorFilterByArea")
		paramsBlob.minArea = Settings.getInstance().getFloatSetting("BlobDetectorMinArea")
		paramsBlob.maxArea = Settings.getInstance().getFloatSetting("BlobDetectorMaxArea")

		paramsBlob.filterByCircularity = Settings.getInstance().getBoolSetting("BlobDetectorFilterByCircularity")
		paramsBlob.minCircularity = Settings.getInstance().getFloatSetting("BlobDetectorMinCircularity")
		paramsBlob.maxCircularity = Settings.getInstance().getFloatSetting("BlobDetectorMaxCircularity")

		paramsBlob.filterByInertia = Settings.getInstance().getBoolSetting("BlobDetectorFilterByInertia")
		paramsBlob.minInertiaRatio = Settings.getInstance().getFloatSetting("BlobDetectorMinInertiaRatio")
		paramsBlob.maxInertiaRatio = Settings.getInstance().getFloatSetting("BlobDetectorMaxInertiaRatio")

		paramsBlob.filterByConvexity = Settings.getInstance().getBoolSetting("BlobDetectorFilterByConvexity")
		paramsBlob.minConvexity = Settings.getInstance().getFloatSetting("BlobDetectorMinConvexity")
		paramsBlob.maxConvexity = Settings.getInstance().getFloatSetting("BlobDetectorMaxConvexity")

		detector = cv.SimpleBlobDetector(paramsBlob)
		cv.vector<cv.KeyPoint> keypoints

		detector.detect(subimage, keypoints)

		dist_min = searchArea * searchArea
		double dist
		for (unsigned i = 0; i < keypoints.size(); i++)
			dist = cv.sqrt((keypoints[i].pt.x - (searchArea + 1)) * (keypoints[i].pt.x - (searchArea + 1)) + (keypoints[i].pt.y - (searchArea + 1)) * (keypoints[i].pt.y - (searchArea + 1)))
			if dist < dist_min:
				point_out.x = off_x + keypoints[i].pt.x
				point_out.y = off_y + keypoints[i].pt.y
				tmp_size = keypoints[i].size
				dist_min = dist


		keypoints.clear()


	subimage.release()

	if size != NULL:
		*size = tmp_size


	return point_out



def detectMarker_threadFinished(self):
	Project.getInstance().getTrials()[m_trial].getMarkers()[m_marker].setSize(m_camera, m_frame, m_size)
	Project.getInstance().getTrials()[m_trial].getMarkers()[m_marker].setPoint(m_camera, m_frame, m_x, m_y, m_refinementAfterTracking ? TRACKED : SET)

	delete m_FutureWatcher
	nbInstances--
	if nbInstances == 0:
		detectMarker_finished.emit()

	delete self


def refinePointPolynomialFit(self, pt, radius_out, darkMarker, camera, trial):
	limmult = 1.6; #multiplies size of box around particle for fitting -- not a sensitive parameter - should be slightly over one... say 1.6
	maskmult = 1; #multiplies fall - off of weighting exponential in fine fit  -- should be about 1.0
	improverthresh = 0.5; #repeat centre refinement cycle if x or y correction is greater than improverthresh
	subpixpeak = True

	double skewness
	double J
	double eccentricity
	double rotation

	radius = radius_out
	x = pt.x
	y = pt.y
#ifdef WRITEIMAGES
	std.cerr << "radius " << radius << std.endl
	std.cerr << "Pt " << x << " " << y << std.endl
#endif
	#repeat subpixel correction until satisfactory
	doextracycles = 3
	stillgood = True
	refinementcount = 0
	maxrefinements = limmult * radius / improverthresh; #if it takes more than maxrefinements to find centre correction, move on

	while (stillgood and (doextracycles > 0))
		refinementcount = refinementcount + 1
		w = (int)(limmult * radius + 0.5)

		#preprocess image
		off_x = (int)(x - w + 0.5)
		off_y = (int)(y - w + 0.5)

		#preprocess image
		cv.Mat subimage
		Project.getInstance().getTrials()[trial].getVideoStreams()[camera].getImage().getSubImage(subimage, w, off_x, off_y)

#ifdef WRITEIMAGES
		cv.imwrite("1_Refine_original.png", subimage)
#endif

		if subimage.cols * subimage.rows < 15:
			#std.cerr << "Too small" << std.endl
			return False


		cv.Mat A
		A.create(subimage.cols * subimage.rows, 15, CV_64F)
		cv.Mat B
		B.create(subimage.cols * subimage.rows, 1, CV_64F)
		cv.Mat p
		p.create(15, 1, CV_64F)

		count = 0
		double tmpx, tmpy, tmpw
		for (j = 0; j < subimage.cols; j++)
			for (i = 0; i < subimage.rows; i++)
				tmpx = off_x - x + j
				tmpy = off_y - y + i
				tmpw = exp(-(tmpx * tmpx + tmpy * tmpy) / (radius * radius * maskmult))
				if darkMarker:
					B.at<double>(count, 0) = tmpw * (255 - subimage.at<uchar>(i, j))

				else:
					B.at<double>(count, 0) = tmpw * (subimage.at<uchar>(i, j))

				A.at<double>(count, 0) = tmpw
				ocol = 1
				for (order = 1; order <= 4; order++)
					for (ocol2 = ocol; ocol2 < ocol + order; ocol2++)
						#std.cerr  << ocol2 << ": Add X to " << ocol2 - order << std.endl
						A.at<double>(count, ocol2) = tmpx * A.at<double>(count, ocol2 - order)

					ocol = ocol + order
					#std.cerr << ocol << ": Add Y to " << ocol - order - 1 << std.endl
					A.at<double>(count, ocol) = tmpy * A.at<double>(count, ocol - order - 1)
					ocol++

				count++



		cv.solve(A, B, p, cv.DECOMP_QR)

		cv.Mat quadric
		quadric.create(subimage.size(), CV_64F)

		double val[6]
		double val2
		for (j = 0; j < subimage.cols; j++)
			for (i = 0; i < subimage.rows; i++)
				tmpx = off_x - x + j
				tmpy = off_y - y + i

				val[0] = 1
				ocol = 1
				for (order = 1; order <= 2; order++)
					for (ocol2 = ocol; ocol2 < ocol + order; ocol2++)
						val[ocol2] = tmpx * val[ocol2 - order]

					ocol = ocol + order
					val[ocol] = tmpy * val[ocol - order - 1]
					ocol++


				val2 = 0
				for (o = 0; o < 6; o++)
					val2 += val[o] * p.at<double>(o, 0)

				quadric.at<double>(i, j) = val2



		a = p.at<double>(3, 0)
		b = p.at<double>(4, 0) / 2.0
		c = p.at<double>(5, 0)
		d = p.at<double>(1, 0) / 2.0
		f = p.at<double>(2, 0) / 2.0
		g = p.at<double>(0, 0)

		J = a * c - b * b
		xc = (b * f - c * d) / J
		yc = (b * d - a * f) / J

		x = x + sign(xc) * min(fabs(xc), improverthresh)
		y = y + sign(yc) * min(fabs(yc), improverthresh)

		rotation = 0.5 * (M_PI / 2.0 - atan((c - a) / 2 / b) + (a - c < 0)) * M_PI / 2.0
		ct = cos(rotation)
		st = sin(rotation)
		P1 = p.at<double>(10, 0) * pow(ct, 4) - p.at<double>(11, 0) * pow(ct, 3) * st + p.at<double>(12, 0) * ct * ct * st * st - p.at<double>(13, 0) * ct * pow(st, 3) + p.at<double>(14, 0) * pow(st, 4)
		P2 = p.at<double>(10, 0) * pow(st, 4) + p.at<double>(11, 0) * pow(st, 3) * ct + p.at<double>(12, 0) * st * st * ct * ct + p.at<double>(13, 0) * st * pow(ct, 3) + p.at<double>(14, 0) * pow(ct, 4)
		Q1 = p.at<double>(3, 0) * ct * ct - p.at<double>(4, 0) * ct * st + p.at<double>(5, 0) * st * st
		Q2 = p.at<double>(3, 0) * st * st + p.at<double>(4, 0) * st * ct + p.at<double>(5, 0) * ct * ct
		radius = fabs(sqrt(sqrt(Q1 * Q2 / P1 / P2 / 36))); #geometric mean

		stillgood = (refinementcount <= maxrefinements) and (J > 0); #if not still good, at once...
		improverswitch = (fabs(xc) > improverthresh) or (fabs(yc) > improverthresh); #check if xc, above thresh or extra cycles required
		doextracycles -= (not improverswitch) ? 1 : 0; #if still good and improverswitch turns off, extra cycles

		if not stillgood or doextracycles == 0:
			semiaxes1 = sqrt(2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g) / -J / ((a - c) * sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - c - a))
			semiaxes2 = sqrt(2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g) / -J / ((c - a) * sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - c - a))
			if semiaxes1 > semiaxes2:
				tmpaxis = semiaxes1
				semiaxes1 = semiaxes2
				semiaxes2 = tmpaxis

			eccentricity = sqrt(1 - (semiaxes1 * semiaxes1) / (semiaxes2 * semiaxes2))

			skewness = (fabs(p.at<double>(6, 0)) + fabs(p.at<double>(7, 0)) + fabs(p.at<double>(8, 0)) + fabs(p.at<double>(9, 0))) * radius / J

			if not subpixpeak:
				#calculate sub - pixel corrections based on centroid of image within particle
				xedge = 2 * radius / sqrt(pow(cos(rotation), 2) + pow(sin(rotation), 2) / (1 - eccentricity * eccentricity)) / (1 + sqrt(1 - eccentricity * eccentricity))
				cv.Mat inparticle
				inparticle.create(quadric.size(), CV_8UC1)
				for (j = 0; j < quadric.cols; j++)
					for (i = 0; i < quadric.rows; i++)
						inparticle.at<char>(i, j) = (quadric.at<double>(i, j) > a * xedge * xedge + 2 * d * xedge + g) ? 1 : 0


				cv.Mat weight
				weight.create(quadric.size(), CV_64F)
				sumweight = 0
				sumx = 0
				sumy = 0
				double tmpWeight
				for (j = 0; j < quadric.cols; j++)
					for (i = 0; i < quadric.rows; i++)
						tmpx = off_x - x + j
						tmpy = off_y - y + i
						tmpw = exp(-(tmpx * tmpx + tmpy * tmpy) / (radius * radius * maskmult))
						tmpWeight = tmpw * inparticle.at<char>(i, j) * subimage.at<char>(i, j)
						weight.at<double>(i, j) = tmpWeight
						sumweight += tmpWeight
						sumx += tmpx * tmpWeight
						sumy += tmpy * tmpWeight


				if sumweight != 0.0:
					xc = sumx / sumweight
					yc = sumx / sumweight


				x = x + xc
				y = y + yc

				weight.release()
				inparticle.release()



		p.release()
		B.release()
		subimage.release()
		A.release()


#ifdef WRITEIMAGES
	std.cerr << "radius " << radius_out << std.endl
	std.cerr << "Pt " << pt.x << " " << pt.y << std.endl
#endif

	if not isnan(radius) and not isnan(x) and not isnan(y):
		pt.y = y
		pt.x = x
		radius_out = radius
		return True
	} else:
		return False



	#std.cerr << "J " << J << std.endl
	#std.cerr << "skewness " << skewness << std.endl
	#std.cerr << "eccentricity " << eccentricity << std.endl
	#std.cerr << "rotation " << rotation << std.endl
	#std.cerr << "radius " << radius << std.endl
	#std.cerr << "Pt " << x << " " << y << std.endl
