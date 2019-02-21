from scipy.spatial import distance
import numpy as np
import imutils
import dlib
import cv2
import time
import math
import re
from math import *
import pandas as pd
from scipy.optimize import curve_fit
from lmfit.models import GaussianModel

class ddestimator:

	#JAY RODGE ===============================================

	TRAINED_MODEL_PATH = './shape_predictor_68_face_landmarks.dat'

	def __init__(self, weights=[1.25,0.5,3,1], purge=True):
		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor(ddestimator.TRAINED_MODEL_PATH)
		self.start_time = int(round(time.time() * 1000))
		self.log = pd.DataFrame(data=[], columns=['ts','key','value'])
		self.log.set_index(['ts', 'key'])
		self.purge=purge
		self.weights=weights
		self.calibration_offset = [0, 0, 0]
		self.has_calibrated = False

	# Used the following code as reference: http://dlib.net/face_landmark_detection.py.html
	def detect_faces(self,  frame, resize_to_width=None, use_gray=True):
		# Faster prediction when frame is resized
		if resize_to_width is not None:
			frame = imutils.resize(frame, width=resize_to_width)
		# If use_gray = True then convert frame used for detection in to grayscale
		if use_gray:
			dframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		else:
			dframe = frame

		#Detect faces in frame
		faces_loc = self.detector(dframe, 0)

		return faces_loc

	def dlib_shape_to_points(self, shape, dtype=np.int32):
		points = np.zeros((68, 2), dtype=dtype)

		for j in range(0, 68):
			points[j] = (shape.part(j).x,shape.part(j).y)

		return points

	def pred_points_on_face(self, frame, face_loc, use_gray=True):
		# If use_gray = True then convert frame used for prediction in to grayscale
		if use_gray:
			pframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		else:
			pframe = frame

		shape = self.predictor(pframe, face_loc)
		points = self.dlib_shape_to_points(shape)
		return points

	def draw_points_on_face(self, frame, points, color):
		for (x, y) in points:
			cv2.circle(frame, (x, y), 1, color, -1)
		return frame

	# SERG MASIS ===============================================

	# These are the estimated 3D positions for 2D image points 17,21,22,26,36,39,42,45,31,35,48,54,57 & 8
	# taken from this model http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp (line 69)
	FACE_3D_ANCHOR_PTS = np.float32([[6.825897, 6.760612, 4.402142],
									 [1.330353, 7.122144, 6.903745],
									 [-1.330353, 7.122144, 6.903745],
									 [-6.825897, 6.760612, 4.402142],
									 [5.311432, 5.485328, 3.987654],
									 [1.789930, 5.393625, 4.413414],
									 [-1.789930, 5.393625, 4.413414],
									 [-5.311432, 5.485328, 3.987654],
									 [2.005628, 1.409845, 6.165652],
									 [-2.005628, 1.409845, 6.165652],
									 [2.774015, -2.080775, 5.048531],
									 [-2.774015, -2.080775, 5.048531],
									 [0.000000, -3.116408, 6.097667],
									 [0.000000, -7.415691, 4.070434]])

	# Retrieved these matrices with OpenCV's camera calibration method
	# method https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
	CAMERA_CALIBRATION_MATRIX = np.float32([[653.0839, 0., 319.5],
											[0., 653.0839, 239.5],
											[0., 0., 1.]])

	#k1, k2, p1, p2, k3
	CAMERA_DISTORTION_COEFFICIENTS = np.float32([[0.070834633684407095],
												 [0.0691402],
												 [0.],
												 [0.],
												 [-1.3073460323689292]])

	# 8 Point Bounding Cube Coordinates
	BOUNDING_CUBE_3D_COORDS = np.float32([[10.0, 10.0, 10.0],
										[10.0, 10.0, -10.0],
										[10.0, -10.0, -10.0],
										[10.0, -10.0, 10.0],
										[-10.0, 10.0, 10.0],
										[-10.0, 10.0, -10.0],
										[-10.0, -10.0, -10.0],
										[-10.0, -10.0, 10.0]])

	def euler_decomposition(self, projmat):
		sin_x = math.sqrt(projmat[0, 0] * projmat[0, 0] + projmat[1, 0] * projmat[1, 0])
		is_singular = sin_x < 0.000001

		if not is_singular:
			x = math.atan2(projmat[2, 1], projmat[2, 2])
			y = math.atan2(-projmat[2, 0], sin_x)
			z = math.atan2(projmat[1, 0], projmat[0, 0])
		else:
			x = math.atan2(-projmat[1, 2], projmat[1, 1])
			y = math.atan2(-projmat[2, 0], sin_x)
			z = 0

		euler = [math.degrees(x), math.degrees(y), math.degrees(z)]
		euler += self.calibration_offset
		euler = np.array(euler).T
		# As explained in this paper: http://www.mirlab.org/conference_papers/international_conference/ICME%202004/html/papers/P38081.pdf
		combined = math.degrees(abs(x) + abs(y) + abs(z))
		#print(str(combined) + " : "+str(euler))
		return euler, combined

	'''
	3D -> 2D point translation from 14/58 'anchor' points of 3D model to 14/68 points of 2D model using this chart (0 indexed)
	33 -> 17 (Left corner left eyebrow)
	29 -> 21 (Right corner left eyebrow)
	34 -> 22 (Left corner right eyebrow)
	38 -> 26 (Right corner right eyebrow)
	13 -> 36 (Left corner left eye)
	17 -> 39 (Right corner left eye)
	25 -> 42 (Left corner right eye)
	21 -> 45 (Left corner right eye)
	55 -> 31 (Left bottom nose)
	49 -> 35 (Right bottom nose)
	43 -> 48 (Left corner mouth)
	39 -> 54 (Right corner mouth)
	45 -> 57 (Bottom center of mouth)
	6 -> 8 (Center of Chin)
	'''
	def est_head_dir(self, points):
		face_2d_anchor_pts = np.array([points[17], points[21], points[22], points[26], points[36], points[39], points[42], points[45], points[31], points[35], points[48], points[54], points[57], points[8]], dtype=np.float32)

		# Get rotation and translation vectors for points and taking in account camera parameters
		_, rotvec, transvec = cv2.solvePnP(ddestimator.FACE_3D_ANCHOR_PTS,
											face_2d_anchor_pts,
											ddestimator.CAMERA_CALIBRATION_MATRIX,
											ddestimator.CAMERA_DISTORTION_COEFFICIENTS)

		# Get rotation matrix with rotation vector
		rotmat, _ = cv2.Rodrigues(rotvec)

		# Get projection matrix by concatenating rotation matrix and translation vector
		projmat = np.hstack((rotmat, transvec))

		# Get Euler angle from projection matrix
		euler, euler_c = self.euler_decomposition(projmat)

		# Set log entries
		self.purge_from_log(3000, 'euler_x')
		self.push_to_log('euler_x', euler[0])
		self.purge_from_log(3000, 'euler_y')
		self.push_to_log('euler_y', euler[1])
		self.purge_from_log(3000, 'euler_z')
		self.push_to_log('euler_z', euler[2])
		self.purge_from_log(3000, 'euler_c')
		self.push_to_log('euler_c', euler_c)

		#print("\t%.2f, %.2f, %.2f, %.2f" % (euler[0], euler[1], euler[2], euler_c))
		return euler, rotvec, transvec

	def get_med_eulers(self, ts_threshold=2000):
		ts = self.get_current_ts() - ts_threshold
		count = self.log[(self.log.ts > ts) & (self.log.key == 'euler_c')]['value'].count()
		if count > round(ts_threshold/200):
			med_x = self.log[(self.log.ts < ts) & (self.log.key == 'euler_x')]['value'].median()
			med_y = self.log[(self.log.ts < ts) & (self.log.key == 'euler_y')]['value'].median()
			med_z = self.log[(self.log.ts < ts) & (self.log.key == 'euler_z')]['value'].median()
			med_c = self.log[(self.log.ts < ts) & (self.log.key == 'euler_c')]['value'].median()
			if not math.isnan(med_x) and not math.isnan(med_y) and not math.isnan(med_z) and not math.isnan(med_c):
				#print("%s: %.2f, %.2f, %.2f, %.2f" % (count, med_c, med_x, med_y, med_z))
				return True, count, np.float32([med_x, med_y, med_z])
			else:
				return False, count, None
		return None, count, None

	def calibrate_camera_angles(self, eulers):
		offsets = eulers * -1
		for i, row in self.log.iterrows():
			if row['key'] == 'euler_x':
				self.log.loc[i, 'value'] += offsets[0]
			elif row['key'] == 'euler_y':
				self.log.loc[i, 'value'] += offsets[1]
			elif row['key'] == 'euler_z':
				self.log.loc[i, 'value'] += offsets[2]
		self.calibration_offset = offsets
		self.has_calibrated = True
		return None

	def est_head_dir_over_time(self, ts_threshold=1000, angle_threshold=45):
		ts = self.get_current_ts() - ts_threshold
		count = self.log[(self.log.ts > ts) & (self.log.key == 'euler_c')]['value'].count()
		if count > round(ts_threshold/200):
			min_x = self.log[(self.log.ts < ts) & (self.log.key == 'euler_x')]['value'].apply(abs).min()
			min_y = self.log[(self.log.ts < ts) & (self.log.key == 'euler_y')]['value'].apply(abs).min()
			min_z = self.log[(self.log.ts < ts) & (self.log.key == 'euler_z')]['value'].apply(abs).min()
			min_c = self.log[(self.log.ts < ts) & (self.log.key == 'euler_c')]['value'].min()
			if not math.isnan(min_x) and not math.isnan(min_y) and not math.isnan(min_z) and not math.isnan(min_c):
				#print("%s: %.2f, %.2f, %.2f, %.2f" % (count, min_c, min_x, min_y, min_z))
				if min_x > angle_threshold or min_y > angle_threshold or min_z > angle_threshold or min_c > angle_threshold:
					ret = True
					self.push_to_log('distracted', self.weights[0])
				else:
					ret = False
					self.push_to_log('distracted', 0)
				return ret, count, np.float32([min_x, min_y, min_z, min_c])
			else:
				return None, count, None
		return None, count, None

	def proj_head_bounding_cube_coords(self, rotation, translation):
		# Project bounding box points using rotation and translation vectors and taking in account camera parameters
		bc_2d_coords, _ = cv2.projectPoints(ddestimator.BOUNDING_CUBE_3D_COORDS,
											rotation,
											translation,
											ddestimator.CAMERA_CALIBRATION_MATRIX,
											ddestimator.CAMERA_DISTORTION_COEFFICIENTS)
		return bc_2d_coords

	def draw_bounding_cube(self, frame, bc_2d_coords, color, euler):
		bc_2d_coords = bc_2d_coords.reshape(8, 2)
		for from_pt, to_pt in np.array([[0, 1], [1, 2], [2, 3], [3, 0],
										[4, 5], [5, 6], [6, 7], [7, 4],
										[0, 4], [1, 5], [2, 6], [3, 7]]):
			cv2.line(frame, tuple(bc_2d_coords[from_pt]), tuple(bc_2d_coords[to_pt]), color)

		label = "({:7.2f}".format(euler[0]) + ",{:7.2f}".format(euler[1]) + ",{:7.2f}".format(euler[2]) + ")"
		cv2.putText(frame, label, tuple(bc_2d_coords[0]),
					cv2.FONT_HERSHEY_PLAIN, 0.65, (0, 0, 0), thickness=1, bottomLeftOrigin=False)
		return frame

	def est_gaze_dir(self, points):
		L_L = (distance.euclidean(points[37], points[36]) + distance.euclidean(points[41], points[36]))/2
		L_R =(distance.euclidean(points[38], points[39]) + distance.euclidean(points[40], points[39]))/2
		R_L = (distance.euclidean(points[43], points[42]) + distance.euclidean(points[47], points[42]))/2
		R_R =(distance.euclidean(points[44], points[45]) + distance.euclidean(points[46], points[45]))/2
		L_ratio = abs((L_L / L_R) - 1)/0.25
		R_ratio = abs((R_L / R_R) - 1)/0.25
		gaze_L = math.degrees(0.3926991 * L_ratio)
		gaze_R = math.degrees(0.3926991 * R_ratio)
		gaze_D = abs(gaze_R - gaze_L)

		# print("==========================")
		# print("L: %.3f <- %.3f = %.3f / %.3f " % (L_ratio,L_L/L_R,L_L,L_R))
		# print("R: %.3f <- %.3f = %.3f / %.3f " % (R_ratio,R_L / R_R, R_L, R_R))
		#print("%.2f - %.2f = %.2f" % (gaze_L, gaze_R, gaze_D))
		self.purge_from_log(3000, 'gaze_L')
		self.push_to_log('gaze_L', gaze_L)
		self.purge_from_log(3000, 'gaze_R')
		self.push_to_log('gaze_R', gaze_R)
		self.purge_from_log(3000, 'gaze_D')
		self.push_to_log('gaze_D', gaze_D)

		if (gaze_L > gaze_R):
			gaze_D = gaze_D * -1
		return gaze_L, gaze_R, gaze_D

	def est_gaze_dir_over_time(self, ts_threshold=2000, angle_threshold=27.5):
		ts = self.get_current_ts() - ts_threshold
		count = self.log[(self.log.ts > ts) & (self.log.key == 'gaze_D')]['value'].count()
		if count > round(ts_threshold/200):
			avg_l = self.log[(self.log.ts < ts) & (self.log.key == 'gaze_L')]['value'].mean()
			avg_r = self.log[(self.log.ts < ts) & (self.log.key == 'gaze_R')]['value'].mean()
			med_d = self.log[(self.log.ts < ts) & (self.log.key == 'gaze_D')]['value'].median()
			if not math.isnan(avg_l) and not math.isnan(avg_r) and not math.isnan(med_d):
				# print("%s: %.2f, %.2f, %.2f" % (count, avg_l, avg_r, med_d))
				if (avg_l > angle_threshold and med_d > (angle_threshold*0.75)) or (avg_r > angle_threshold and med_d > (angle_threshold*0.75)):
					ret = True
					self.push_to_log('distracted', self.weights[1])
				else:
					ret = False
					self.push_to_log('distracted', 0)
				return ret, count, np.float32([avg_l, avg_r, med_d])
		return None, count, None

	def proj_gaze_line_coords(self, rotation, translation, gaze_D):
		d = 10
		z = 6.763430
		x = d * math.tan(math.radians(abs(gaze_D)))
		if gaze_D < 0:
			x = x * -1
		gl_3d_coords = np.float32([[0.0, 0.0, z],[x, 0.0, z + d]])
		gl_2d_coords,_ = cv2.projectPoints(gl_3d_coords,
											rotation,
											translation,
											ddestimator.CAMERA_CALIBRATION_MATRIX,
											ddestimator.CAMERA_DISTORTION_COEFFICIENTS)
		return gl_2d_coords

	def draw_gaze_line(self, frame, gl_2d_coords, color, gaze_D):
		gl_2d_coords = gl_2d_coords.reshape(2, 2)
		for from_pt, to_pt in np.array([[0, 1]]):
			cv2.line(frame, tuple(gl_2d_coords[from_pt]), tuple(gl_2d_coords[to_pt]), color)

		cv2.putText(frame, "{:7.2f}".format(gaze_D), tuple(gl_2d_coords[1]),
					cv2.FONT_HERSHEY_PLAIN, 0.65, (0, 0, 0), thickness=1, bottomLeftOrigin=False)
		return frame

	def calc_kss(self, ts_threshold=10000):
		ts = self.get_current_ts() - ts_threshold
		count = self.log[(self.log.ts > ts) & ((self.log.key == 'distracted') | (self.log.key == 'drowsiness'))]['value'].count()
		if count > round(ts_threshold/200):
			sum_distracted = self.log[(self.log.ts > ts) & (self.log.key == 'distracted')]['value'].sum()
			sum_drowsy = self.log[(self.log.ts > ts) & (self.log.key == 'drowsiness')]['value'].sum()
			sum = sum_distracted + sum_drowsy
			#print("\t%.2f = %.2f + %.2f" % (sum, sum_distracted, sum_drowsy))
			if not math.isnan(sum):
				kss = sum / count
				self.push_to_log('kss', kss)
				if kss > 10:
					kss = 10
				return kss
			else:
				return 0
		else:
			return None

	def create_progress_bar(self, width, height, percentage=0, status=None):
		if percentage > 1:
			percentage = 1
		elif percentage < 0:
			percentage = 0
		image = np.zeros((height, width, 3), np.uint8)
		size = int((width - 16) * percentage)
		cv2.rectangle(image, (6, 6), (width - 6, height - 6), (0, 255, 0), 1)
		if size > 0:
			cv2.rectangle(image, (9, 9), (9 + size, height - 9), (0, 255, 0), cv2.FILLED)
		if status is not None:
			cv2.putText(image, status, (15, height - 13), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), thickness=3)
			cv2.putText(image, status, (15, height - 13), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), thickness=1)
		return image

	def draw_progress_bar(self, frame, width, height, percentage=0, status=""):
		progressbar = self.create_progress_bar(width, height, percentage, status)
		y_offset = frame.shape[0] - height
		x_offset = 0
		frame[y_offset:y_offset + progressbar.shape[0], x_offset:x_offset + progressbar.shape[1]] = progressbar
		return frame

	def get_current_ts(self):
		ts = int(round(time.time() * 1000)) - self.start_time
		return ts

	def push_to_log(self, key, value):
		ts = self.get_current_ts()
		self.log = self.log.append({'ts': ts, 'key':key, 'value':value}, ignore_index=True)
		return self.log['ts'].count()

	def purge_from_log(self, ts_threshold, key):
		if self.purge:
			ts = self.get_current_ts() - ts_threshold
			self.log = self.log.drop(self.log[(self.log.ts < ts) & (self.log.key == key)].index)
		return self.log['ts'].count()

	def fetch_log(self, key=None, ts_threshold=None):
		log = None
		if ts_threshold is None:
			if key is None:
				log = self.log
			else:
				log = self.log[self.log.key == key]
		else:
			ts = self.get_current_ts() - ts_threshold
			if key is None:
				log = self.log[(self.log.ts < ts)]
			else:
				log = self.log[(self.log.ts < ts) & (self.log.key == key)]
		return log

	# JAY RODGE ===============================================
	def est_eye_openness(self, points):
		ear_L = self.get_ear(points[42:48])
		ear_R = self.get_ear(points[36:42])
		ear_B = (ear_L + ear_R) / 2
		#print("%.2f + %.2f / 2 = %.2f" % (ear_L, ear_R, ear_B))

		self.purge_from_log(3000, 'ear_L')
		self.push_to_log('ear_L', ear_L)
		self.purge_from_log(3000, 'ear_R')
		self.push_to_log('ear_R', ear_R)
		self.purge_from_log(3000, 'ear_B')
		self.push_to_log('ear_B', ear_B)

		return ear_B, ear_R, ear_L

	def get_ear(self, eye_points):
		A = distance.euclidean(eye_points[1], eye_points[5]) #p2-p6
		B = distance.euclidean(eye_points[2], eye_points[4]) #p3-p5
		C = distance.euclidean(eye_points[0], eye_points[3]) #p1-p4
		ear = (A + B) / (2.0 * C)
		return ear

	def draw_eye_lines(self, frame, points, ear_R, ear_L):
		leftEye = points[42:48]
		rightEye = points[36:42]
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1) 
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)
		cv2.putText(frame, "{:.2f}".format(ear_L), tuple([points[41][0]-5,points[41][1]+10]),
					cv2.FONT_HERSHEY_PLAIN, 0.75, (0, 0, 0), thickness=1)
		cv2.putText(frame, "{:.2f}".format(ear_R), tuple([points[47][0]-10,points[47][1]+10]),
					cv2.FONT_HERSHEY_PLAIN, 0.75, (0, 0, 0), thickness=1)
		return frame

	def get_eye_closedness_over_time(self, ts_threshold=1000, ear_threshold=0.25):
		ts = self.get_current_ts() - ts_threshold
		df_ear_b = self.log[(self.log.ts > ts) & (self.log.key == 'ear_B')]
		count = df_ear_b['value'].count()
		if count > round(ts_threshold/200):
			avg = df_ear_b['value'].mean()
			max = df_ear_b['value'].max()
			#print("%.3f , %.3f < %.2f" % (avg, max, ear_threshold))
			if max < ear_threshold:
				ret = True
				closed = True
				self.push_to_log('drowsiness', self.weights[2])
			elif avg < ear_threshold:
				ret = True
				closed = False
				self.push_to_log('drowsiness', round(self.weights[2]/2,1))
			else:
				ret = closed = False
				self.push_to_log('drowsiness', 0)
			return ret, count, np.float32([max, avg]), closed
		return None, count, None, None


	def est_mouth_openess(self, points):
		lip_top_bottom = distance.euclidean(points[51], points[57])
		mouth_top_bottom = distance.euclidean(points[62], points[66])
		top_bottom = np.mean([lip_top_bottom, mouth_top_bottom])
		lip_left_right = distance.euclidean(points[48], points[54])
		mouth_left_right = distance.euclidean(points[60], points[64])
		left_right = np.mean([lip_left_right, mouth_left_right])
		mouth_ratio = top_bottom / left_right
		self.purge_from_log(10000, 'mar')
		self.push_to_log('mar', mouth_ratio)
		return mouth_ratio

	def draw_mouth(self, frame, points, mar):
		mouth_points = points[48:59]
		mouthHull = cv2.convexHull(mouth_points)
		cv2.drawContours(frame, [mouthHull],-1, (0, 0, 255), 1)
		cv2.putText(frame, "{:.2f}".format(mar), tuple([points[57][0]-15,points[57][1]+10]),
					cv2.FONT_HERSHEY_PLAIN, 0.75, (0, 0, 0), thickness=1)
		return frame

	def get_mouth_openess_over_time(self, ts_threshold=4750, mar_threshold=0.6):
		ts = self.get_current_ts() - ts_threshold
		df_mar = self.log[(self.log.ts > ts) & (self.log.key == 'mar')]
		count = df_mar['value'].count()
		if count > round(ts_threshold/200):
			max_val = df_mar['value'].max()
			if max_val > mar_threshold:
				min_ts = df_mar['ts'].min()
				max_ts = df_mar['ts'].max()
				ts_4th = (max_ts - min_ts) / 4
				q1_ts = round(min_ts + ts_4th)
				q3_ts = round(max_ts - ts_4th)
				#print(df_mar['value'].idxmax())
				max_val_ts = df_mar.loc[df_mar['value'].idxmax(), 'ts']
				#print("%.2f < %.2f < %.2f" % (q1_ts,max_val_ts,q3_ts))
				if max_val_ts > q1_ts and max_val_ts < q3_ts:
					x = np.array(df_mar['ts'].tolist())
					y = np.array(df_mar['value'].tolist())
					mse_model, mse_yhat, _, _, _ = self.fit_to_gaussian(x, y)
					#print("\t%.2f\t%.2f" % (mse_model, mse_yhat))
					if mse_model > 0.2 and mse_yhat < 0.1:
						self.push_to_log('drowsiness', math.ceil(self.weights[3]*(mse_model*2)))
						return True, count, mse_model, mse_yhat
			self.push_to_log('drowsiness', 0)
			return False, count, None, None
		return None, count, None, None

	#Used this as reference: https://lmfit.github.io/lmfit-py/model.html#lmfit.model.Model.fit
	def fit_to_gaussian(self, x, y):
		gmodel = GaussianModel()
		params = gmodel.guess(y, x=x)
		c = params['center'].value
		n = len(y)
		q3 = ((np.max(x) - c) / 2) + c
		min_x = np.min(x)
		q1 = ((params['center'].value - min_x) / 2) + min_x
		s = params['sigma'].value
		h = params['height'].value
		max_y = np.max(y)
		if np.max([h, max_y]) < 0.5:
			amp = 1 / n
			diff_h = 0.6 - h
			gmodel.set_param_hint('amplitude', value=amp)
			gmodel.set_param_hint('amplitude', max=amp * (1 + diff_h))
			gmodel.set_param_hint('amplitude', min=amp * diff_h)
		gmodel.set_param_hint('center', value=c)
		gmodel.set_param_hint('center', max=q3)
		gmodel.set_param_hint('center', min=q1)
		gmodel.set_param_hint('sigma', value=s)
		gmodel.set_param_hint('sigma', min=s / 2)
		gmodel.set_param_hint('sigma', max=s * 1.5)
		gmodel.set_param_hint('height', min=0.6)
		result = gmodel.fit(y, x=x)
		# gmodel.print_param_hints()
		report = result.fit_report()
		chi_re = re.compile(r'chi-square\s+=\s+([0-9.]+)')
		cor_re = re.compile(r'C\(sigma, amplitude\)\s+=\s+([0-9.-]+)')
		chis = np.float32(chi_re.findall(report))
		cors = np.float32(cor_re.findall(report))
		coeffs = np.concatenate((chis, cors))
		mse_model = self.assess_fit(y, result.init_fit - result.best_fit)
		mse_yhat = self.assess_fit(y, result.residual)
		return mse_model, mse_yhat, result, report, coeffs

	def assess_fit(self, y, residuals):
		n = len(y)
		RSS = 0
		for i in range(0, n):
			RSS += residuals[i] ** 2
		MSE = RSS / n
		return MSE
