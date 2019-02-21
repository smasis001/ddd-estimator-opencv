import imutils
import cv2
import sys
import os
import time
import pyximport; pyximport.install()
import ddestimator
import pandas as pd
import tkinter as tk
from tkinter import messagebox

class demo2:

	FRAME_WIDTH = 500
	WINDOW_TITLE = "Demo #2: distraction and drowsiness estimation on a file"

	PROCESS_INTERVAL = 50

	K_ESC = 27
	K_QUIT = ord('q')
	K_POINTS = ord('p')
	K_BOUNDING = ord('b')
	K_GAZE = ord('g')
	K_EYES = ord('e')
	K_MOUTH = ord('m')
	K_DD = ord('d')
	K_NONE = ord('n')
	K_REFRESH = ord('r')
	K_SAVE_LOG = ord('l')
	K_HELP = ord('h')

	LOG_PATH = '%d/%f_kss_%ts.csv'

	CALIBRATE_CAMERA_ANGLES = True

	def __init__(self):
		if len(sys.argv) == 2:
			self.videofilepath = sys.argv[1]
			self.videodirpath = os.path.dirname(self.videofilepath)
			self.videofilename = os.path.basename(self.videofilepath)
		else:
			self.videofilepath = self.videodirpath = self.videofilename = None

		self.rootwin = tk.Tk()
		self.rootwin.withdraw()
		cv2.namedWindow(demo2.WINDOW_TITLE)
		self.show_points = False
		self.show_bounding = False
		self.show_gaze = False
		self.show_ear = False
		self.show_mar = False
		self.show_dd = True
		self.ddestimator = ddestimator.ddestimator()
		#self.ddestimator = ctypes.cdll.LoadLibrary('./ddestimator.cpython-36m-darwin.so')
		self.has_calibrated = False

	def run(self):
		if self.videofilepath is None:
			sys.exit("Location of valid video file is required")

		self.cap = cv2.VideoCapture(self.videofilepath)
		if not self.cap.isOpened():
			print("Unable to read file.")
			return
		while self.cap.isOpened() and self.cap.get(cv2.CAP_PROP_POS_FRAMES) < self.cap.get(cv2.CAP_PROP_FRAME_COUNT):
			self.key_strokes_handler()
			ret, frame = self.cap.read()
			if ret:
				#print("\t%s / %s" % (self.cap.get(cv2.CAP_PROP_POS_FRAMES), self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
				#frame = imutils.resize(frame, width=demo2.FRAME_WIDTH)
				#TODO: Remove this for production
				#if self.cap.get(cv2.CAP_PROP_POS_FRAMES) % 3 == 0:
				frame = self.process_frame(frame)
				h = frame.shape[0]
				w = frame.shape[1]
				pos = str(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))) + "/" + str(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
				frame = cv2.putText(frame, pos, (w - 120, h - 30),
									cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), thickness=1)
				cv2.imshow(demo2.WINDOW_TITLE, frame)
				#cv2.moveWindow(demo2.WINDOW_TITLE, 0, 0)

		kss_log = self.ddestimator.fetch_log('kss')
		ts = int(round(time.time() * 1000))
		path = (demo2.LOG_PATH).replace('%ts', str(ts)).replace('%d', self.videodirpath).replace('%f', self.videofilename)
		print("\t"+path)
		kss_log.to_csv(path)

	def process_frame(self, frame=None):

		#Detect faces in frame
		faces_loc = self.ddestimator.detect_faces(frame, None, True)

		#If there's more than one face...
		if len(faces_loc) > 0:

			#Only interested in first face found (for his demo)
			face_loc = faces_loc[0]

			#Predict coordinates of 68 points of this face using ML trained model
			points = self.ddestimator.pred_points_on_face(frame, face_loc)

			#All immediate estimations based on points locations
			euler, rotation, translation = self.ddestimator.est_head_dir(points)
			#- Calibrate for camera angles based on euler angles
			if demo2.CALIBRATE_CAMERA_ANGLES and not self.ddestimator.has_calibrated:
				has_calibration, _, meds = self.ddestimator.get_med_eulers()
				if has_calibration:
					self.ddestimator.calibrate_camera_angles(meds)
			_, _, gaze_D = self.ddestimator.est_gaze_dir(points)
			ear_B, ear_R, ear_L = self.ddestimator.est_eye_openness(points)
			mar = self.ddestimator.est_mouth_openess(points)

			#All timescale estimations based on points locations
			head_distraction, _, _ = self.ddestimator.est_head_dir_over_time()
			if not head_distraction:
				gaze_distraction, _, _ = self.ddestimator.est_gaze_dir_over_time()
			else:
				gaze_distraction = False
			eye_drowsiness, _, _, eye_closedness = self.ddestimator.get_eye_closedness_over_time()
			did_yawn, _, _, _ = self.ddestimator.get_mouth_openess_over_time()

			#Calc KSS with previous measurements
			kss = self.ddestimator.calc_kss()
			if kss is not None:
				print("\t%.2f" % (kss*10))

			#Show results on frame
			if self.show_points:
				frame = self.ddestimator.draw_points_on_face(frame, points, (0, 0, 255))

			if self.show_bounding:
				bc_2d_coords = self.ddestimator.proj_head_bounding_cube_coords(rotation, translation)
				frame = self.ddestimator.draw_bounding_cube(frame, bc_2d_coords, (0, 0, 255), euler)

			if self.show_gaze:
				gl_2d_coords = self.ddestimator.proj_gaze_line_coords(rotation, translation, gaze_D)
				self.ddestimator.draw_gaze_line(frame, gl_2d_coords, (0, 255, 0), gaze_D)

			if self.show_ear:
				frame = self.ddestimator.draw_eye_lines(frame, points, ear_R, ear_L)

			if self.show_mar:
				frame = self.ddestimator.draw_mouth(frame, points, mar)

			if self.show_dd:
				h = frame.shape[0]
				if head_distraction:
					cv2.putText(frame, "DISTRACTED [h]", (20, h - 100),
								cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 0), thickness=1)
				elif self.show_gaze and gaze_distraction:
					cv2.putText(frame, "distracted [g]", (20, h - 100),
								cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 0), thickness=1)
				if did_yawn:
					cv2.putText(frame, "DROWSY [y]", (20, h - 80),
								cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 255), thickness=1)

				if eye_closedness:
					cv2.putText(frame, "DROWSY [ec]", (20, h - 60),
								cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 255), thickness=1)
				elif self.show_ear and eye_drowsiness:
					cv2.putText(frame, "drowsy [ed]", (20, h - 60),
								cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 0), thickness=1)

				if kss is not None:
					kss_int = int(round(kss*10))
					frame = self.ddestimator.draw_progress_bar(frame, 140, 35, kss, str(kss_int))

		return frame

	def key_strokes_handler(self):
		pressed_key = cv2.waitKey(1) & 0xFF

		if pressed_key == demo2.K_ESC or pressed_key == demo2.K_QUIT:
			print('-> QUIT')
			self.cap.release()
			cv2.destroyAllWindows()
			sys.exit(0)

		elif pressed_key == demo2.K_POINTS:
			print('-> SHOW FACIAL LANDMARKS')
			self.show_points = True
			return None

		elif pressed_key == demo2.K_BOUNDING:
			print('-> SHOW BOUNDING CUBE FOR HEAD DIRECTION ESTIMATION')
			self.show_bounding = True
			return None

		elif pressed_key == demo2.K_GAZE:
			print('-> SHOW LINES FOR GAZE DIRECTION ESTIMATION')
			self.show_gaze = True
			return None

		elif pressed_key == demo2.K_EYES:
			print('-> SHOW EYE OPENNESS ESTIMATION')
			self.show_ear = True
			return None

		elif pressed_key == demo2.K_MOUTH:
			print('-> SHOW MOUTH OPENNESS ESTIMATION')
			self.show_mar = True
			return None

		elif pressed_key == demo2.K_DD:
			print('-> SHOW DROWSINESS & DISTRACTION ESTIMATIONS')
			self.show_dd = True
			return None

		elif pressed_key == demo2.K_NONE:
			print('-> SHOW NO ESTIMATIONS')
			self.show_bounding = False
			self.show_gaze = False
			self.show_ear = False
			self.show_mar = False
			self.show_dd = False
			self.show_points = False
			return None

		elif pressed_key == demo2.K_REFRESH:
			print('-> RESET SHOW TO DEFAULT')
			self.show_bounding = False
			self.show_gaze = False
			self.show_ear = False
			self.show_mar = False
			self.show_dd = True
			self.show_points = False
			return None

		elif pressed_key == demo2.K_SAVE_LOG:
			print('-> SAVE LOG FILE WITH KSS ESTIMATIONS')
			kss_log = self.ddestimator.fetch_log('kss')
			ts = int(round(time.time() * 1000))
			path = (demo2.LOG_PATH).replace('%ts', str(ts)).replace('%d', self.videodirpath).replace('%f', self.videofilename)
			print("\t"+path)
			kss_log.to_csv(path)
			return None

		# TODO: help screen
		elif pressed_key == demo2.K_HELP:
			tk.messagebox.showinfo("Help",
			                       "'p': Show facial landmarks\r\n'b': Show bounding cube\r\n'g': Show gaze line\r\n'e': Show eye info\r\n'm': Show mouth info\r\n'd': Show drowsiness & distraction info\r\n'n': Show nothing\r\n'r': Refresh/clear the frame of all info\r\n'l': Save log file\r\n'q': Quit the program")
			return None

		else:
			return None

if __name__ == '__main__':
	demo2 = demo2()
	demo2.run()