import os
import os.path as osp
import numpy as np
import torch
import cv2
import json
import copy
import math
import random
from glob import glob
from pycocotools.coco import COCO
from config import cfg
from utils.mano import mano
from utils.smpl_x import smpl_x
from utils.preprocessing import load_img, set_aspect_ratio, augmentation
from utils.vis import vis_kpt
from pytorch3d.io import load_ply

class HIC(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        assert data_split == 'test', 'only testing is supported for HIC dataset'
        self.data_path = osp.join('..', 'data', 'HIC', 'data')
        self.focal = (525.0, 525.0)
        self.princpt = (319.5, 239.5)
        self.datalist = self.load_data()
        
    def load_data(self):
        # load annotation
        db = COCO(osp.join(self.data_path, 'HIC.json'))

        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_width, img_height = img['width'], img['height']
            img_path = osp.join(self.data_path, img['file_name'])
            hand_type = ann['hand_type']

            # bbox
            bbox = np.array([0, 0, img_width, img_height], dtype=np.float32)
            bbox = set_aspect_ratio(bbox, cfg.input_img_shape[1]/cfg.input_img_shape[0])

            # mano mesh
            if ann['right_mano_path'] is not None:
                right_mano_path = osp.join(self.data_path, ann['right_mano_path'])
            else:
                right_mano_path = None
            if ann['left_mano_path'] is not None:
                left_mano_path = osp.join(self.data_path, ann['left_mano_path'])
            else:
                left_mano_path = None

            datalist.append({
                'aid': aid,
                'img_path': img_path,
                'img_shape': (img_height, img_width),
                'bbox': bbox,
                'hand_type': hand_type,
                'right_mano_path': right_mano_path,
                'left_mano_path': left_mano_path})
            
        return datalist
    
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox = data['img_path'], data['img_shape'], data['bbox']

        # img
        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
        img = self.transform(img.astype(np.float32))/255.

        # mano coordinates
        right_mano_path = data['right_mano_path']
        if right_mano_path is not None:
            rmano_vert_cam, _ = load_ply(right_mano_path)
        else:
            rmano_vert_cam = torch.zeros((mano.vertex_num, 3)).float()
        left_mano_path = data['left_mano_path']
        if left_mano_path is not None:
            lmano_vert_cam, _ = load_ply(left_mano_path)
        else:
            lmano_vert_cam = torch.zeros((mano.vertex_num, 3)).float()
        
        inputs = {'img': img}
        targets = {'rmano_vert_cam': rmano_vert_cam, 'lmano_vert_cam': lmano_vert_cam}
        meta_info = {}
        return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        eval_result = {
                    'mpvpe_sh': [None for _ in range(sample_num)],
                    'mpvpe_ih': [None for _ in range(sample_num*2)],
                    'rrve': [None for _ in range(sample_num)],
                    'mrrpe': [None for _ in range(sample_num)]
                    }

        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            out = outs[n]

            rhand_vert_out = out['smplx_vert_cam'][smpl_x.hand_vertex_idx['right_hand'],:] * 1000 # meter to millimeter
            lhand_vert_out = out['smplx_vert_cam'][smpl_x.hand_vertex_idx['left_hand'],:] * 1000 # meter to millimeter
            rhand_vert_gt = out['rmano_vert_cam_target'] * 1000 # meter to milimeter
            lhand_vert_gt = out['lmano_vert_cam_target'] * 1000 # meter to milimeter
            rhand_root_out = np.dot(mano.kpt['regressor'], rhand_vert_out)[mano.kpt['root_idx']]
            lhand_root_out = np.dot(mano.kpt['regressor'], lhand_vert_out)[mano.kpt['root_idx']]
            rhand_root_gt = np.dot(mano.kpt['regressor'], rhand_vert_gt)[mano.kpt['root_idx']]
            lhand_root_gt = np.dot(mano.kpt['regressor'], lhand_vert_gt)[mano.kpt['root_idx']]
          
            # mrrpe
            rel_trans_gt = lhand_root_gt - rhand_root_gt
            rel_trans_out = lhand_root_out - rhand_root_out
            if annot['hand_type'] == 'interacting':
                eval_result['mrrpe'][n] = np.sqrt(np.sum((rel_trans_gt - rel_trans_out)**2))

            # mpvpe (right hand relative)
            if annot['hand_type'] == 'interacting':
                if annot['right_mano_path'] is not None and annot['left_mano_path'] is not None:
                    _rhand_vert_out = rhand_vert_out - rhand_root_out[None]
                    _lhand_vert_out = lhand_vert_out - rhand_root_out[None]
                    _rhand_vert_gt = rhand_vert_gt - rhand_root_gt[None]
                    _lhand_vert_gt = lhand_vert_gt - rhand_root_gt[None]
                    vert_out = np.concatenate((_rhand_vert_out, _lhand_vert_out))
                    vert_gt = np.concatenate((_rhand_vert_gt, _lhand_vert_gt))
                    eval_result['rrve'][n] = np.sqrt(np.sum((vert_out - vert_gt)**2,1)).mean()

            # mpvpe
            if annot['hand_type'] == 'right' and annot['right_mano_path'] is not None:
                vert_out = rhand_vert_out - rhand_root_out[None]
                vert_gt = rhand_vert_gt - rhand_root_gt[None]
                eval_result['mpvpe_sh'][n] = np.sqrt(np.sum((vert_out - vert_gt)**2,1)).mean()
            elif annot['hand_type'] == 'left' and annot['left_mano_path'] is not None:
                vert_out = lhand_vert_out - lhand_root_out[None]
                vert_gt = lhand_vert_gt - lhand_root_gt[None]
                eval_result['mpvpe_sh'][n] = np.sqrt(np.sum((vert_out - vert_gt)**2,1)).mean()
            elif annot['hand_type'] == 'interacting':
                if annot['right_mano_path'] is not None:
                    vert_out = rhand_vert_out - rhand_root_out[None]
                    vert_gt = rhand_vert_gt - rhand_root_gt[None]
                    eval_result['mpvpe_ih'][2*n] = np.sqrt(np.sum((vert_out - vert_gt)**2,1)).mean()
                if annot['left_mano_path'] is not None:
                    vert_out = lhand_vert_out - lhand_root_out[None]
                    vert_gt = lhand_vert_gt - lhand_root_gt[None]
                    eval_result['mpvpe_ih'][2*n+1] = np.sqrt(np.sum((vert_out - vert_gt)**2,1)).mean()

        return eval_result
    
    def print_eval_result(self, eval_result):
        tot_eval_result = {
                'mpvpe_sh': [],
                'mpvpe_ih': [],
                'rrve': [],
                'mrrpe': []
                }
        
        # mpvpe (average all samples)
        for mpvpe_sh in eval_result['mpvpe_sh']:
            if mpvpe_sh is not None:
                tot_eval_result['mpvpe_sh'].append(mpvpe_sh)
        for mpvpe_ih in eval_result['mpvpe_ih']:
            if mpvpe_ih is not None:
                tot_eval_result['mpvpe_ih'].append(mpvpe_ih)
        for rrve in eval_result['rrve']:
            if rrve is not None:
                tot_eval_result['rrve'].append(rrve)
       
        # mrrpe (average all samples)
        for mrrpe in eval_result['mrrpe']:
            if mrrpe is not None:
                tot_eval_result['mrrpe'].append(mrrpe)
 
        # print evaluation results
        eval_result = tot_eval_result
        
        print('MRRPE: %.2f mm' % (np.mean(eval_result['mrrpe'])))
        print('MPVPE for all hand sequences: %.2f mm' % (np.mean(eval_result['mpvpe_sh'] + eval_result['mpvpe_ih'])))
        print('MPVPE for single hand sequences: %.2f mm' % (np.mean(eval_result['mpvpe_sh'])))
        print('MPVPE for interacting hand sequences: %.2f mm' % (np.mean(eval_result['mpvpe_ih'])))
        print('RRVE for interacting hand sequences: %.2f mm' % (np.mean(eval_result['rrve'])))
