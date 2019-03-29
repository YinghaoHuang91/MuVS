'''
Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this computer program. 

You can only use this computer program if you have closed a license agreement with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.

Any use of the computer program without a valid license is prohibited and liable to prosecution.

Copyright©2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics. All rights reserved. 

Contact: ps-license@tuebingen.mpg.de
'''

import os
import pickle as pkl
import scipy.io as sio
import numpy as np
from camera import Perspective_Camera
import cv2

# TODO: choose gender
GENDER = 'f'
HEVA_PATH = '../Data/HEVA_Validate'
SMPL_PATH = '../Data/Smpl_Model/basicModel_%s_lbs_10_207_0_v1.0.0.pkl' % GENDER
# TODO: set beta dimension
N_BETAS = 10
SMPL_JOINT_IDS = [8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20, 12]
TORSO_IDS = [3-1, 4-1, 9-1, 10-1]
HEAD_VID = 411
# TODO
NUM_VIEW = 3
# TODO
VIS_OR_NOT = False
# TODO
LOG_OR_NOT = 1
# TODO
BATCH_FRAME_NUM = 30
# TODO
DCT_NUM = 10
DCT_MAT_PATH = '../Data/DCT_Basis/%d.mat' % BATCH_FRAME_NUM
# TODO
#tem_j2d_ids = [0, 1, 4, 5, 6, 7, 10, 11]
#tem_smpl_joint_ids = [8, 5, 4, 7, 21, 19, 18, 20]
TEM_J2D_IDS = range(0, 13)
TEM_SMPL_JOINT_IDS = SMPL_JOINT_IDS
# TODO
POSE_PRIOR_PATH = '../Data/Prior/genericPrior.mat'


def load_data(img_path, num_view):
	imgs = []
	j2ds = []
	segs = []
	cams = []
	
	for i in range(1, num_view + 1):
		# load image
		img_i_path = img_path.replace('_C1', '_C' + str(i))
		img_i = cv2.imread(img_i_path)
		imgs.append(img_i)
	
		# load 2d joint
		j2d_i_path = img_i_path.replace('Image', 'Pose_2D')
		j2d_i_path = j2d_i_path.replace('.png', '.png_pose.npz')
		j2d_i = np.load(j2d_i_path)
		#j2d_i = j2d_i['pose'].T[:, :3]
		j2d_i = j2d_i['pose'].T[:, :2]

		j2ds.append(j2d_i)

		# load segmentation
		# TODO
		'''
		seg_i_path = img_i_path.replace('Image', 'Segmentation')
		seg_i_path = seg_i_path.replace('.png', '.png_segmentation.npz_vis.png')
		seg_i = cv2.imread(seg_i_path)
		seg_i = cv2.split(seg_i)[0]
		seg_i[seg_i > 0] = 1
		segs.append(seg_i)
		'''
		segs.append(None)

		# load camera
		cam_i_path = img_i_path.replace('Image', 'GT')
		cam_i_path = os.path.join(os.path.dirname(cam_i_path), 'camera.mat')
		cam_i = sio.loadmat(cam_i_path, squeeze_me=True, struct_as_record=False)
		cam_i = cam_i['camera']

		cam = Perspective_Camera(cam_i.focal_length[0], cam_i.focal_length[1], cam_i.principal_pt[0], cam_i.principal_pt[1],
					cam_i.t / 1000.0, cam_i.R_angles)
		cams.append(cam)

	#j2ds = np.concatenate(j2ds, axis=0)
	j2ds = np.array(j2ds)
	return imgs, j2ds, segs, cams
	

def load_data_temporal(img_files):
	imgs = []; j2ds = []; poses = []; 
	betas = []; trans = []
	
	
	for img_f in img_files:
		img_i, j2d_i_tmp, _, cam_i = load_data(img_f, NUM_VIEW)	
		j2d_i = j2d_i_tmp[:, TEM_J2D_IDS]
		imgs.append(img_i); j2ds.append(j2d_i); 

		pose_path = img_f.replace('Image', 'Res_1')
		extension = os.path.splitext(pose_path)[1]
		pose_path = pose_path.replace(extension, '.pkl')

		with open(pose_path, 'rb') as fin:
			res_1 = pkl.load(fin)
		poses.append( np.array(res_1['pose']) )
		betas.append( res_1['betas'] )
		trans.append( np.array(res_1['trans']) )

	mean_betas = np.array(betas)
	mean_betas = np.mean(mean_betas, axis=0)

	return imgs, j2ds, cam_i, poses, mean_betas, trans

def load_dct_base():
	mtx = sio.loadmat(DCT_MAT_PATH, squeeze_me=True, struct_as_record=False)	
	mtx = mtx['D']
	mtx = mtx[:DCT_NUM]

	return np.array(mtx)

def load_initial_param():
	pose_prior = sio.loadmat(POSE_PRIOR_PATH, squeeze_me=True, struct_as_record=False)
	pose_mean = pose_prior['mean']
	pose_covariance = np.linalg.inv(pose_prior['covariance'])
	zero_shape = np.ones([13]) * 1e-8 # extra 3 for zero global rotation
	zero_trans = np.ones([3]) * 1e-8
	initial_param = np.concatenate([zero_shape, pose_mean, zero_trans], axis=0)

	return initial_param, pose_mean, pose_covariance
