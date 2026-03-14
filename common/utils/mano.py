import numpy as np
import torch
import os.path as osp
from config import cfg
import smplx
from utils.transforms import change_kpt_name

class MANO(object):
    def __init__(self):
        self.layer_arg = {'create_global_orient': False, 'create_hand_pose': False, 'create_betas': False, 'create_transl': False}
        self.layer = {'right': smplx.create(cfg.human_model_path, 'mano', is_rhand=True, use_pca=False, flat_hand_mean=False, **self.layer_arg), 'left': smplx.create(cfg.human_model_path, 'mano', is_rhand=False, use_pca=False, flat_hand_mean=False, **self.layer_arg)}
        self.vertex_num = 778
        self.face = {'right': self.layer['right'].faces, 'left': self.layer['left'].faces}
        self.shape_param_dim = 10
        self.is_boundary = self.get_hand_boundary()

        # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
        if torch.sum(torch.abs(self.layer['left'].shapedirs[:,0,:] - self.layer['right'].shapedirs[:,0,:])) < 1:
            print('Fix shapedirs bug of MANO')
            self.layer['left'].shapedirs[:,0,:] *= -1

        # joint (single hand)
        self.joint = {
                'num': 16, 
                'name': ('Wrist', 'Index_1', 'Index_2', 'Index_3', 'Middle_1', 'Middle_2', 'Middle_3', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Ring_1', 'Ring_2', 'Ring_3', 'Thumb_1', 'Thumb_2', 'Thumb_3'),
                'flip_pair': ()
                }
        self.joint['root_idx'] = self.joint['name'].index('Wrist')

        # keypoint (single hand)
        self.kpt = {
                'num': 21, 
                'name': ('Wrist', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_1', 'Index_2', 'Index_3', 'Index_4', 'Middle_1', 'Middle_2', 'Middle_3', 'Middle_4', 'Ring_1', 'Ring_2', 'Ring_3', 'Ring_4', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Pinky_4'),
                'flip_pair': ()
                }
        self.kpt['root_idx'] = self.kpt['name'].index('Wrist')
        self.kpt['regressor'] = change_kpt_name(self.layer['right'].J_regressor.numpy(), self.joint['name'], self.kpt['name']) # same for the right and left hands
        self.kpt['regressor'][self.kpt['name'].index('Thumb_4')] = np.array([1 if i == 745 else 0 for i in range(self.vertex_num)], dtype=np.float32)
        self.kpt['regressor'][self.kpt['name'].index('Index_4')] = np.array([1 if i == 317 else 0 for i in range(self.vertex_num)], dtype=np.float32)
        self.kpt['regressor'][self.kpt['name'].index('Middle_4')] = np.array([1 if i == 445 else 0 for i in range(self.vertex_num)], dtype=np.float32)
        self.kpt['regressor'][self.kpt['name'].index('Ring_4')] = np.array([1 if i == 556 else 0 for i in range(self.vertex_num)], dtype=np.float32)
        self.kpt['regressor'][self.kpt['name'].index('Pinky_4')] = np.array([1 if i == 673 else 0 for i in range(self.vertex_num)], dtype=np.float32)
        self.kpt['rigid_align_idx'] = [i for i in range(self.kpt['num']) if self.kpt['name'][i] in ('Wrist', 'Index_1', 'Middle_1', 'Ring_1', 'Pinky_1')]

        # keypoint (two hands)
        self.kpt_th = {
                'num': 42, 
                'name': ('R_Wrist', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4', 'L_Wrist', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4'),
                'flip_pairs': [ (i,i+21) for i in range(21)]
                }
        self.kpt_th['part_idx'] = {'right': np.arange(0,self.kpt_th['num']//2), 'left': np.arange(self.kpt_th['num']//2,self.kpt_th['num'])}
        self.kpt_th['root_idx'] = {'right': self.kpt_th['name'].index('R_Wrist'), 'left': self.kpt_th['name'].index('L_Wrist')}

    def get_hand_boundary(self):
        # detect boundary region (around the wrist)
        vertex_masks = {}
        for h in ('right', 'left'):
            cnt = np.zeros((self.vertex_num, self.vertex_num), dtype=np.float32)
            for f in self.face[h]:
                v1, v2, v3 = f
                cnt[v1,v2] += 1
                cnt[v2,v1] += 1

                cnt[v1,v3] += 1
                cnt[v3,v1] += 1

                cnt[v2,v3] += 1
                cnt[v3,v2] += 1
            cnt[np.triu_indices_from(cnt)] = 0 # remove duplicated edges
            y, x = np.where(cnt == 1)

            # vertex mask
            vertex_mask = np.zeros((self.vertex_num), dtype=np.float32)
            vertex_mask[x] = 1
            vertex_mask[y] = 1
            vertex_masks[h] = vertex_mask
        
        return vertex_masks

mano = MANO()
