import numpy as np
import torch
import os.path as osp
from config import cfg
import smplx

class SMPL(object):
    def __init__(self):
        self.layer_arg = {'create_body_pose': False, 'create_betas': False, 'create_global_orient': False, 'create_transl': False}
        self.layer = {'neutral': smplx.create(cfg.human_model_path, 'smpl', gender='NEUTRAL', **self.layer_arg), 'male': smplx.create(cfg.human_model_path, 'smpl', gender='MALE', **self.layer_arg), 'female': smplx.create(cfg.human_model_path, 'smpl', gender='FEMALE', **self.layer_arg)}
        self.vertex_num = 6890
        self.face = self.layer['neutral'].faces.astype(np.int64)
        self.shape_param_dim = 10
        
        # joint
        self.joint = {
                'num': 24,
                'name': ('Pelvis', 'L_Hip', 'R_Hip', 'Spine_1', 'L_Knee', 'R_Knee', 'Spine_2', 'L_Ankle', 'R_Ankle', 'Spine_3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'),
                'flip_pair': ( (1,2), (4,5), (7,8), (10,11), (13,14), (16,17), (18,19), (20,21), (22,23) )
                }
        self.joint['root_idx'] = self.joint['name'].index('Pelvis')
        self.vert_to_joint = self.layer['neutral'].J_regressor.numpy().astype(np.float32)

        # keypoint
        self.kpt = self.joint
        
smpl = SMPL()
