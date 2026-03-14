import numpy as np
import torch
import os.path as osp
from config import cfg
import smplx
import pickle
from utils.mano import mano

class SMPLX(object):
    def __init__(self):
        self.layer_arg = {'create_global_orient': False, 'create_body_pose': False, 'create_left_hand_pose': False, 'create_right_hand_pose': False, 'create_jaw_pose': False, 'create_leye_pose': False, 'create_reye_pose': False, 'create_betas': False, 'create_expression': False, 'create_transl': False}
        self.shape_param_dim = 10
        self.expr_param_dim = 10
        self.layer = {'neutral': smplx.create(cfg.human_model_path, 'smplx', gender='NEUTRAL', num_betas=self.shape_param_dim, use_pca=False, use_face_contour=True, **self.layer_arg),
                        'male': smplx.create(cfg.human_model_path, 'smplx', gender='MALE', num_betas=self.shape_param_dim, use_pca=False, use_face_contour=True, **self.layer_arg),
                        'female': smplx.create(cfg.human_model_path, 'smplx', gender='FEMALE', num_betas=self.shape_param_dim, use_pca=False, use_face_contour=True, **self.layer_arg)
                        }
        self.vertex_num = 10475
        self.face = self.layer['neutral'].faces.astype(np.int64)
        with open(osp.join(cfg.human_model_path, 'smplx', 'MANO_SMPLX_vertex_ids.pkl'), 'rb') as f:
            self.hand_vertex_idx = pickle.load(f, encoding='latin1')
        self.face_vertex_idx = np.load(osp.join(cfg.human_model_path, 'smplx', 'SMPL-X__FLAME_vertex_ids.npy'))
        self.vert_neighbor_idxs = self.get_vert_neighbor()
        self.hand_boundary_idx = {'right': self.hand_vertex_idx['right_hand'][mano.is_boundary['right']==1], 'left': self.hand_vertex_idx['left_hand'][mano.is_boundary['left']==1]}
        self.vert_to_joint = self.layer['neutral'].J_regressor.numpy()
        self.vert_to_joint_idx = {'pelvis': 0, 'lwrist': 20, 'rwrist': 21, 'neck': 12}
        with open(osp.join(cfg.human_model_path, 'smplx', 'SMPLX_to_J14.pkl'), 'rb') as f:
            self.vert_to_joint14 = pickle.load(f, encoding='latin1')
        self.vert_to_hand_joint = self.make_hand_regressor()

        # joint
        self.joint = {
                'num': 55, # 22 (body joints) + 3 (face joints) + 30 (hand joints)
                'name': ('Pelvis', 'L_Hip', 'R_Hip', 'Spine_1', 'L_Knee', 'R_Knee', 'Spine_2', 'L_Ankle', 'R_Ankle', 'Spine_3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', # body joints
                        'Jaw', 'L_Eye', 'R_Eye', # face joints
                        'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', # left hand joints
                        'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3' # right hand joints
                        ),
                'flip_pair': ( (1,2), (4,5), (7,8), (10,11), (13,14), (16,17), (18,19), (20,21), # body joints
                            (23, 24), # face joints
                            (25,40), (26,41), (27,42), (28,43), (29,44), (30,45), (31,46), (32,47), (33,48), (34,49), (35,50), (36,51), (37,52), (38,53), (39,54) # hand joints
                            )
                        }
        self.joint['root_idx'] = self.joint['name'].index('Pelvis')
        self.joint['part_idx'] = {'body': range(self.joint['name'].index('Pelvis'), self.joint['name'].index('R_Wrist')+1),
                                'face': range(self.joint['name'].index('Jaw'), self.joint['name'].index('R_Eye')+1),
                                'lhand': range(self.joint['name'].index('L_Index_1'), self.joint['name'].index('L_Thumb_3')+1),
                                'rhand': range(self.joint['name'].index('R_Index_1'), self.joint['name'].index('R_Thumb_3')+1)
                                }
        
        # keypoint
        self.kpt = {
                'num': 137, # 25 (body kpts) + 40 (hand kpts) + 72 (face kpts)
                'name': ('Pelvis', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Neck', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', 'L_Ear', 'R_Ear', 'L_Eye', 'R_Eye', 'Nose', # body kpts
                         'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4', # left hand kpts
                         'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4', # right hand kpts
                         *['Face_' + str(i) for i in range(1,73)] # face keypoints (too many keypoints... omit real names. have same name of keypoints defined in FLAME class)
                         ),
                'flip_pair': ( (1,2), (3,4), (5,6), (8,9), (10,11), (12,13), (14,17), (15,18), (16,19), (20,21), (22,23), # body kpts
                        (25,45), (26,46), (27,47), (28,48), (29,49), (30,50), (31,51), (32,52), (33,53), (34,54), (35,55), (36,56), (37,57), (38,58), (39,59), (40,60), (41,61), (42,62), (43,63), (44,64), # hand kpts
                        (67,68), # face eyeballs
                        (69,78), (70,77), (71,76), (72,75), (73,74), # face eyebrow
                        (83,87), (84,86), # face below nose
                        (88,97), (89,96), (90,95), (91,94), (92,99), (93,98), # face eyes
                        (100,106), (101,105), (102,104), (107,111), (108,110), # face mouth
                        (112,116), (113,115), (117,119), # face lip
                        (120,136), (121,135), (122,134), (123,133), (124,132), (125,131), (126,130), (127,129) # face contours
                        ),
                'idx': (0,1,2,4,5,7,8,12,16,17,18,19,20,21,60,61,62,63,64,65,59,58,57,56,55, # body kpts
                        37,38,39,66,25,26,27,67,28,29,30,68,34,35,36,69,31,32,33,70, # left hand kpts
                        52,53,54,71,40,41,42,72,43,44,45,73,49,50,51,74,46,47,48,75, # right hand kpts
                        22,15, # jaw, head
                        57,56, # eyeballs
                        76,77,78,79,80,81,82,83,84,85, # eyebrow
                        86,87,88,89, # nose
                        90,91,92,93,94, # below nose
                        95,96,97,98,99,100,101,102,103,104,105,106, # eyes
                        107, # right mouth
                        108,109,110,111,112, # upper mouth
                        113, # left mouth
                        114,115,116,117,118, # lower mouth
                        119, # right lip
                        120,121,122, # upper lip
                        123, # left lip
                        124,125,126, # lower lip
                        127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143 # face contour
                        )
                }
        self.kpt['root_idx'] = self.kpt['name'].index('Pelvis')
        self.kpt['part_idx'] = {'body': range(self.kpt['name'].index('Pelvis'), self.kpt['name'].index('Nose')+1),
                            'lhand': range(self.kpt['name'].index('L_Thumb_1'), self.kpt['name'].index('L_Pinky_4')+1),
                            'rhand': range(self.kpt['name'].index('R_Thumb_1'), self.kpt['name'].index('R_Pinky_4')+1),
                            'face': range(self.kpt['name'].index('Face_1'), self.kpt['name'].index('Face_72')+1)}
        self.kpt['hand_rigid_align_idx'] = {'right': [i for i in range(self.kpt['num']) if self.kpt['name'][i] in ('R_Wrist', 'R_Index_1', 'R_Middle_1', 'R_Ring_1', 'R_Pinky_1')],
                                            'left': [i for i in range(self.kpt['num']) if self.kpt['name'][i] in ('L_Wrist', 'L_Index_1', 'L_Middle_1', 'L_Ring_1', 'L_Pinky_1')]}
        
        # keypoint used for the heatmap representation
        self.kpt_hm = {
                'num': 65, # 25 (body kpts) + 40 (hand kpts)
                'name': ('Pelvis', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Neck', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', 'L_Ear', 'R_Ear', 'L_Eye', 'R_Eye', 'Nose', # body kpts
                         'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4', # left hand kpts
                         'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4' # right hand kpts
                         )
                    }
        self.kpt_hm['root_idx'] = self.kpt_hm['name'].index('Pelvis')
        self.kpt_hm['part_idx'] = {'body': range(self.kpt_hm['name'].index('Pelvis'), self.kpt_hm['name'].index('Nose')+1),
                                'lhand': range(self.kpt_hm['name'].index('L_Thumb_1'), self.kpt_hm['name'].index('L_Pinky_4')+1),
                                'rhand': range(self.kpt_hm['name'].index('R_Thumb_1'), self.kpt_hm['name'].index('R_Pinky_4')+1),
                                'face': range(self.kpt_hm['name'].index('L_Ear'), self.kpt_hm['name'].index('Nose')+1)}
        
    
    def kpt_to_kpt_hm(self, kpt):
        kpt_hm = []
        for name in self.kpt_hm['name']:
            idx = self.kpt['name'].index(name)
            kpt_hm.append(kpt[:,idx,:])
        kpt_hm = torch.stack(kpt_hm,1)
        return kpt_hm
   
    def make_hand_regressor(self):
        regressor = self.layer['neutral'].J_regressor.numpy()
        lhand_regressor = np.concatenate((regressor[[20,37,38,39],:],
                                            np.eye(self.vertex_num)[5361,None],
                                                regressor[[25,26,27],:],
                                                np.eye(self.vertex_num)[4933,None],
                                                regressor[[28,29,30],:],
                                                np.eye(self.vertex_num)[5058,None],
                                                regressor[[34,35,36],:],
                                                np.eye(self.vertex_num)[5169,None],
                                                regressor[[31,32,33],:],
                                                np.eye(self.vertex_num)[5286,None]))
        rhand_regressor = np.concatenate((regressor[[21,52,53,54],:],
                                            np.eye(self.vertex_num)[8079,None],
                                                regressor[[40,41,42],:],
                                                np.eye(self.vertex_num)[7669,None],
                                                regressor[[43,44,45],:],
                                                np.eye(self.vertex_num)[7794,None],
                                                regressor[[49,50,51],:],
                                                np.eye(self.vertex_num)[7905,None],
                                                regressor[[46,47,48],:],
                                                np.eye(self.vertex_num)[8022,None]))
        hand_regressor = {'left': lhand_regressor, 'right': rhand_regressor}
        return hand_regressor

    def get_vert_neighbor(self, neighbor_max_num=10):
        vertex_num, face = self.vertex_num, self.face

        adj = {i: set() for i in range(vertex_num)}
        for i in range(len(face)):
            for idx in face[i]:
                adj[idx] |= set(face[i]) - set([idx])

        neighbor_idxs = np.ones((vertex_num, neighbor_max_num), dtype=np.float32) * -1
        for idx in range(vertex_num):
            neighbor_num = min(len(adj[idx]), neighbor_max_num)
            neighbor_idxs[idx,:neighbor_num] = np.array(list(adj[idx]))[:neighbor_num]

        neighbor_idxs = torch.from_numpy(neighbor_idxs)
        return neighbor_idxs

smpl_x = SMPLX()
