# MuVS
This is the demo code for the MuVS (Multi-view SMPLify) method presented in the paper:

[Yinghao Huang, Federica Bogo, Christoph Lassner, Angjoo Kanazawa, Peter V. Gehler, Javier Romero, Ijaz Akhter and Michael J. Black, Towards Accurate Marker-less Human Shape and Pose Estimation over Time, International Conference on 3D Vision (3DV) 2017](https://ps.is.tuebingen.mpg.de/publications/muvs-3dv-2017).


### Dependencies

You need to install the the following packages via pip like shown below:
```
pip install --upgrade numpy scipy tensorflow-gpu matplotlib
pip install git+https://github.com/mattloper/chumpy.git@refs/pull/18/head # required since the main branch is broken since pip 10
pip install https://github.com/MPI-IS/meshlite/raw/master/download/psbody_meshlite-0.1-cp27-cp27mu-linux_x86_64.whl # meshlite
```
Additionally, smpl needs to be downloaed from http://smpl.is.tue.mpg.de/downloads and symlinked 

```
ln -s <smpl dir>/smpl_webuser TF_Code/smpl

```

[Deepcut](https://github.com/eldar/deepcut-cnn) and one [human-specific segmentation](https://github.com/classner/up) method are also used (Not used now, since no differentialbe render in Tensorflow availabe). You need to install them from the project links.

### Folder structure
#### Code_TF
This is where the code resides. A brief summary of what each file does in the following:
```
camera.py: perspective pinhole camera model, which projects 3D points into image space.
smpl_batch.py: Tensorflow verion of SMPL model, which takes in SMPL paramters and yields 3D SMPL joints.
util.py: experimental settings and data loading methods.
```

#### Data
Image data, DCT basis, SMPL model and pose prior is stored here. If you want to try your own data, you need to prepare it in the way shown in folder *HEVA_Validate*:
```
1, One folder for each view.
2, For each view, images are in *Image* subfolder.
3, 2D pose information in *Pose_2D subfolder*.
4, Create subfolders *Res_1* and *Res_2* to save the results.
```


### How to run

Firstly you have to organize your data like described in the previous setction. Then you need to run Deepercut for all the images, and place them in the right folder. After that you can use these commands to run MuVS from the main folder:

```
./gen_run_jobs_1.sh # This runs per-frame fitting for all the frames
./gen_run_jobs_2.sh # This runs temporal fitting for all the temporal units
```

The resultant files (3D SMPL mesh, SMPL shape, pose and translation parameters) will be saved in ply and pkl in Res_1/Res_2 respectively.


### Citation
If you find this code useful for your research, please consider citing:
```
@inproceedings{MuVS:3DV:2017,
  title = {Towards Accurate Marker-less Human Shape and Pose Estimation over Time},
  author = {Huang, Yinghao and Bogo, Federica and Lassner, Christoph and Kanazawa, Angjoo and Gehler, Peter V. and Romero, Javier and Akhter, Ijaz and Black, Michael J.},
  booktitle = {International Conference on 3D Vision (3DV)},
  year = {2017}
}
```                                                                                                                                                                    
