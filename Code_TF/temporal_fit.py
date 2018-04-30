import cv2
import os
import numpy as np
import scipy.io as sio
import pickle as pkl
import tensorflow as tf
import glob
import util
import matplotlib.pyplot as plt
from smpl_batch import SMPL
from tensorflow.contrib.opt import ScipyOptimizerInterface as scipy_pt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('data_prefix', type=str)
parser.add_argument('start_idx', type=int, help='index starts from 0')
args = parser.parse_args()

def main(img_files):
	imgs, j2ds, cams, poses, mean_betas, trans = util.load_data_temporal(img_files)
	j2ds = np.array(j2ds).reshape([-1, 2])
	dct_mtx = util.load_dct_base()
	dct_mtx = tf.constant(dct_mtx.T, dtype=tf.float32)

	# For SMPL parameters
	params_tem = []
	params_pose_tem = []
	param_shape = tf.constant(mean_betas, dtype=tf.float32)
	for idx in range(0, util.BATCH_FRAME_NUM):
		param_pose = tf.Variable(poses[idx], dtype=tf.float32, name='Pose_%d' % idx)
		param_trans = tf.constant(trans[idx], dtype=tf.float32)
		param = tf.concat([param_shape, param_pose, param_trans], axis=1)

		params_tem.append(param)
		params_pose_tem.append(param_pose)
	params_tem = tf.concat(params_tem, axis=0)
	
	# For DCT prior params
	c_dct = tf.Variable(np.zeros([len(util.TEM_SMPL_JOINT_IDS), 3, util.DCT_NUM]), dtype=tf.float32, name='C_DCT')
	smpl_model = SMPL(util.SMPL_PATH)	
	
	j3ds, vs = smpl_model.get_3d_joints(params_tem, util.TEM_SMPL_JOINT_IDS) # N x M x 3
	j3ds = j3ds[:, :-1]
	j3ds_flatten = tf.reshape(j3ds, [-1, 3])
	j2ds_est = []
	for idx in range(0, util.NUM_VIEW):
		tmp = cams[idx].project(j3ds_flatten)
		j2ds_est.append(tmp)
	j2ds_est = tf.concat(j2ds_est, axis=0)
	j2ds_est = tf.reshape(j2ds_est, [util.NUM_VIEW, util.BATCH_FRAME_NUM, len(util.TEM_SMPL_JOINT_IDS), 2])
	j2ds_est = tf.transpose(j2ds_est, [1, 0, 2, 3])
	j2ds_est = tf.reshape(j2ds_est, [-1, 2])
	
	_, pose_mean, pose_covariance = util.load_initial_param()
	pose_mean = tf.constant(pose_mean, dtype=tf.float32)
	pose_covariance = tf.constant(pose_covariance, dtype=tf.float32)

	objs = {}
	objs['J2D_Loss'] = tf.reduce_sum( tf.square(j2ds_est - j2ds) )
	for i in range(0, util.BATCH_FRAME_NUM):
		pose_diff = params_pose_tem[i][:, -69:] - pose_mean
		objs['Prior_Loss_%d' % i] = 5 * tf.squeeze( tf.matmul(tf.matmul(pose_diff, pose_covariance), tf.transpose(pose_diff)) ) 

	for i, jid in enumerate(util.TEM_SMPL_JOINT_IDS):
		for j, aid in enumerate([0, 1, 2]):
		#for j, aid in enumerate([0, 2]):
			trajectory = j3ds[:, i, aid]	
			'''
			c_dct_initial = tf.matmul(tf.expand_dims(trajectory, axis=0), dct_mtx)
			c_dct_initial = tf.squeeze(c_dct_initial)
			'''
			
			#import ipdb; ipdb.set_trace()
			#with tf.control_dependencies( [tf.assign(c_dct[i, j], c_dct_initial)] ):
			trajectory_dct = tf.matmul(dct_mtx, tf.expand_dims(c_dct[i, j], axis=-1))
			trajectory_dct = tf.squeeze(trajectory_dct)

			objs['DCT_%d_%d' % (i, j)] = tf.reduce_sum( tf.square(trajectory - trajectory_dct) )
	loss = tf.reduce_mean(objs.values())

	if util.VIS_OR_NOT:
		func_callback = on_step
	else:
		func_callback = None

		
	sess = tf.Session()
        sess.run(tf.global_variables_initializer())

	def lc(j2d_est):
		_, ax = plt.subplots(1, 3)
		for idx in range(0, util.NUM_VIEW):
			import copy
			tmp = copy.copy(imgs[idx])
			for j2d in j2ds[idx]:
				x = int( j2d[1] )
				y = int( j2d[0] )

				if x > imgs[0].shape[0] or x > imgs[0].shape[1]:
					continue
				tmp[x:x+5, y:y+5, :] = np.array([0, 0, 255])

			for j2d in j2d_est[idx]:
				x = int( j2d[1] )
				y = int( j2d[0] )

				if x > imgs[0].shape[0] or x > imgs[0].shape[1]:
					continue
				tmp[x:x+5, y:y+5, :] = np.array([255, 0, 0])
			ax[idx].imshow(tmp)
		plt.show()

	if util.VIS_OR_NOT:
		func_lc = None
	else:
		func_lc = None

        optimizer = scipy_pt(loss=loss, var_list=params_pose_tem + [c_dct], options={'ftol':0.001, 'maxiter':500, 'disp':True}, method='L-BFGS-B')
        #optimizer.minimize(sess, fetches = [objs], loss_callback=func_lc)
        optimizer.minimize(sess, loss_callback=func_lc)
		
	print sess.run(c_dct)
		
	vs_final = sess.run(vs)
	pose_final = sess.run(params_pose_tem)
	betas = sess.run(param_shape)
	
	model_f = sess.run(smpl_model.f)
	model_f = model_f.astype(int).tolist()

	for fid in range(0, util.BATCH_FRAME_NUM / 2):
		from psbody.meshlite import Mesh
		m = Mesh(v=vs_final[fid], f=model_f)
		out_ply_path = img_files[fid].replace('Image', 'Res_2')
		extension = os.path.splitext(out_ply_path)[1]
		out_ply_path = out_ply_path.replace(extension, '.ply')
		m.write_ply(out_ply_path)

		res = {'pose': pose_final[fid], 'betas': betas, 'trans': trans[fid]}
		out_pkl_path = out_ply_path.replace('.ply', '.pkl')

		print out_pkl_path
		with open(out_pkl_path, 'wb') as fout:
			pkl.dump(res, fout)

		

if __name__ == '__main__':
	data_prefix = args.data_prefix	
	start_idx = args.start_idx * util.BATCH_FRAME_NUM

	img_files = glob.glob(os.path.join(util.HEVA_PATH, data_prefix + '_1_C1', 'Image', '*.png'))
	# For the last one
	if (start_idx + util.BATCH_FRAME_NUM) > len(img_files):
		start_idx = len(img_files) - 30
	img_files = img_files[start_idx  : (start_idx + util.BATCH_FRAME_NUM)]

	main(img_files)

