import tensorflow as tf
import numpy as np
import cv2
from psbody.meshlite import Mesh

class Perspective_Camera():
	# Input is in numpy array format
	def __init__(self, focal_length_x, focal_length_y, center_x, center_y, trans, axis_angle):
		self.fl_x = tf.constant(focal_length_x, dtype=tf.float32)
		self.fl_y = tf.constant(focal_length_y, dtype=tf.float32)
		self.cx = tf.constant(center_x, dtype=tf.float32)
		self.cy = tf.constant(center_y, dtype=tf.float32)
		self.trans = tf.constant(trans, dtype=tf.float32)
		self.rotm = tf.constant(cv2.Rodrigues(axis_angle)[0], dtype=tf.float32)

	# points: Nx3
	def transform(self, points):
		#points = points + tf.reshape(self.trans, [1, 3])
		points = tf.expand_dims(points, axis=-1)
		rotm = tf.tile(tf.reshape(self.rotm, [1, 3, 3]), [tf.shape(points)[0], 1, 1])
		points = tf.matmul(rotm, points)
		points = tf.squeeze(points)
		res = points + tf.reshape(self.trans, [1, 3])

		return res
		

	# Point is a Tensor
	def project(self, points):
		points = self.transform(points)
		points = points + 1e-8

		xs = tf.divide(points[:, 0], points[:, 2])
		ys = tf.divide(points[:, 1], points[:, 2])
		us = self.fl_x * xs + self.cx
		vs = self.fl_y * ys + self.cy
		#vs = 480 - vs
		res = tf.stack([us, vs], axis=1)

		return res


if __name__ == '__main__':
	m = Mesh()
	m.load_from_ply('/ps/scratch/yhuang/ESMPLify_4_12_Public/Data/HEVA_Validate/S1_Box_1_C1/Res_1/frame0010.ply')
	import cv2
	img = cv2.imread('/ps/scratch/yhuang/ESMPLify_4_12_Public/Data/HEVA_Validate/S1_Box_1_C1/Image/frame0010.png')
	import scipy.io as sio
	cam_data = sio.loadmat('/ps/scratch/yhuang/ESMPLify_4_12_Public/Data/HEVA_Validate/S1_Box_1_C1/GT/camera.mat', squeeze_me=True, struct_as_record=False)
	cam_data = cam_data['camera']
	cam = Perspective_Camera(cam_data.focal_length[0], cam_data.focal_length[1], cam_data.principal_pt[0], cam_data.principal_pt[1],
					cam_data.t / 1000.0, cam_data.R_angles)
	v = tf.constant(m.v, dtype=tf.float32)
	j2ds = cam.project(v)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	j2ds = sess.run(j2ds)


	import ipdb; ipdb.set_trace()
	for p in j2ds:
		x = int(p[0])
		y = int(p[1])
		if x < 0 or y < 0:
			continue
		if x < img.shape[0] and y < img.shape[1]:
			img[x, y, :] = 0
	cv2.imshow('Img', img)
	cv2.waitKey(0)
