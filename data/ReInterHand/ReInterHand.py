import os
import os.path as osp
import numpy as np
import torch
import cv2
import json
import copy
from pycocotools.coco import COCO
from config import cfg
from utils.mano import mano
from utils.smpl_x import smpl_x
from utils.preprocessing import load_img, get_bbox, set_aspect_ratio, augmentation, process_kpt, process_mano_param
from utils.transforms import change_kpt_name
from utils.vis import vis_kpt, render_mesh
from glob import glob

class ReInterHand(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.data_path = osp.join('..', 'data', 'ReInterHand', 'data')
        self.envmap_mode = 'envmap_per_frame'
        self.test_capture_ids = ['m--20221215--0949--RNS217--pilot--ProjectGoliathScript--Hands--two-hands', 'm--20221216--0953--NKC880--pilot--ProjectGoliathScript--Hands--two-hands', 'm--20230317--1130--QZX685--pilot--ProjectGoliath--Hands--two-hands']
        self.datalist = self.load_data()
        
    def load_data(self):
        datalist = []
        capture_id_list = [x.split('/')[-1] for x in glob(osp.join(self.data_path, '*')) if osp.isdir(x)]
        for capture_id in capture_id_list:
            if self.data_split == 'train':
                if capture_id in self.test_capture_ids:
                    continue
            else:
                if capture_id not in self.test_capture_ids:
                    continue
            
            with open(osp.join(self.data_path, capture_id, 'Mugsy_cameras', 'cam_params.json')) as f:
                cam_param = json.load(f)
                for cam_name in cam_param.keys():
                    cam_param[cam_name] = {k: np.array(v, dtype=np.float32) for k,v in cam_param[cam_name].items()}
            for cam_name in cam_param.keys():
                frame_idx_list = [int(x.split('/')[-1][:-4]) for x in glob(osp.join(self.data_path, capture_id, 'Mugsy_cameras', self.envmap_mode, 'images', cam_name, '*'))]
                for frame_idx in frame_idx_list:
                    img_path = osp.join(self.data_path, capture_id, 'Mugsy_cameras', self.envmap_mode, 'images', cam_name, '%06d.png' % frame_idx)
                    rhand_mano_param_path = osp.join(self.data_path, capture_id, 'mano_fits', 'params', str(frame_idx) + '_right.json')
                    lhand_mano_param_path = osp.join(self.data_path, capture_id, 'mano_fits', 'params', str(frame_idx) + '_left.json')

                    datalist.append({
                        'capture_id': capture_id,
                        'cam_name': cam_name,
                        'frame_idx': frame_idx,
                        'img_path': img_path,
                        'rhand_mano_param_path': rhand_mano_param_path,
                        'lhand_mano_param_path': lhand_mano_param_path,
                        'cam_param': cam_param[cam_name]})
            
        return datalist
    
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        capture_id, frame_idx = data['capture_id'], data['frame_idx']
        img_path, rhand_mano_param_path, lhand_mano_param_path, cam_param = data['img_path'], data['rhand_mano_param_path'], data['lhand_mano_param_path'], data['cam_param']
        cam_param['t'] /= 1000 # millimeter -> meter

        # img
        img = load_img(img_path)
        img_height, img_width = img.shape[:2]
        img_shape = (img_height, img_width)
        bbox = np.array([0, 0, img_width, img_height], dtype=np.float32)
        bbox = set_aspect_ratio(bbox, cfg.input_img_shape[1]/cfg.input_img_shape[0]) 
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
        img = self.transform(img.astype(np.float32))/255.

        if self.data_split == 'train':
            # mano coordinates (right hand)
            with open(rhand_mano_param_path) as f:
                mano_param = json.load(f)
            mano_param['hand_type'] = 'right'
            rmano_root_pose, rmano_pose, rmano_shape_param, rmano_kpt_cam, rmano_kpt_img, _ = process_mano_param(mano_param, cam_param, do_flip, img_shape, rot)
            rmano_root_valid = float(True)
            rmano_pose_valid = np.ones((mano.joint['num'],1), dtype=np.float32)
            rmano_shape_valid = float(True)
            rmano_kpt_valid = np.ones((mano.kpt['num'],1), dtype=np.float32)

            # mano coordinates (left hand)
            with open(lhand_mano_param_path) as f:
                mano_param = json.load(f)
            mano_param['hand_type'] = 'left'
            lmano_root_pose, lmano_pose, lmano_shape_param, lmano_kpt_cam, lmano_kpt_img, _ = process_mano_param(mano_param, cam_param, do_flip, img_shape, rot)
            lmano_root_valid = float(True)
            lmano_pose_valid = np.ones((mano.joint['num'],1), dtype=np.float32)
            lmano_shape_valid = float(True)
            lmano_kpt_valid = np.ones((mano.kpt['num'],1), dtype=np.float32)

            # change name when flip
            if do_flip:
                rmano_root_pose, lmano_root_pose = lmano_root_pose, rmano_root_pose
                rmano_pose, lmano_pose = lmano_pose, rmano_pose
                rmano_shape_param, lmano_shape_param = lmano_shape_param, rmano_shape_param
                rmano_kpt_cam, lmano_kpt_cam = lmano_kpt_cam, rmano_kpt_cam
                rmano_kpt_img, lmano_kpt_img = lmano_kpt_img, rmano_kpt_img
                rmano_root_valid, lmano_root_valid = lmano_root_valid, rmano_root_valid
                rmano_pose_valid, lmano_pose_valid = lmano_pose_valid, rmano_pose_valid
                rmano_shape_valid, lmano_shape_valid = lmano_shape_valid, rmano_shape_valid
                rmano_kpt_valid, lmano_kpt_valid = lmano_kpt_valid, rmano_kpt_valid

            # aggregate two-hand data
            mano_pose = np.concatenate((rmano_pose, lmano_pose))
            mano_kpt_cam = np.concatenate((rmano_kpt_cam, lmano_kpt_cam)) # follow mano.kpt_th['name']
            mano_kpt_img = np.concatenate((rmano_kpt_img, lmano_kpt_img)) # follow mano.kpt_th['name']
            mano_pose_valid = np.concatenate((rmano_pose_valid, lmano_pose_valid))
            mano_kpt_valid = np.concatenate((rmano_kpt_valid, lmano_kpt_valid)) # follow mano.kpt_th['name']
     
            # apply data augmentation to the mano coordinates
            # flip is already applied
            mano_kpt_cam = mano_kpt_cam - mano_kpt_cam[mano.kpt_th['root_idx']['right'],None,:] # right hand root-relative
            smplx_kpt_img, smplx_kpt_cam, smplx_kpt_valid, smplx_kpt_trunc = process_kpt(mano_kpt_img, mano_kpt_cam, mano_kpt_valid, False, img_shape, None, img2bb_trans, rot, mano.kpt_th['name'], smpl_x.kpt['name'])
           
            # change to smplx
            mano_pose_name = ['R_' + name for name in mano.joint['name'] if name != 'Wrist'] + ['L_' + name for name in mano.joint['name'] if name != 'Wrist']
            smplx_pose = change_kpt_name(mano_pose, mano_pose_name, smpl_x.joint['name'])
            smplx_pose_valid = change_kpt_name(mano_pose_valid, mano_pose_name, smpl_x.joint['name'])
            smplx_shape = np.zeros((smpl_x.shape_param_dim), dtype=np.float32)
            smplx_shape_valid = float(False)
            smplx_expr = np.zeros((smpl_x.expr_param_dim), dtype=np.float32)
            smplx_expr_valid = float(False)
            
            """
            # for debug
            _tmp = smplx_kpt_img.copy()
            _tmp[:,0] = _tmp[:,0] / cfg.vit_output_shape[1] * cfg.input_img_shape[1]
            _tmp[:,1] = _tmp[:,1] / cfg.vit_output_shape[0] * cfg.input_img_shape[0]
            _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
            _img = vis_kpt(_img.copy(), _tmp)
            cv2.imwrite('reih_' + str(idx) + '_mano.jpg', _img)
            print('saved')
            """
            
            inputs = {'img': img}
            targets = {'kpt_img': smplx_kpt_img, 'smplx_kpt_img': smplx_kpt_img, 'kpt_cam': smplx_kpt_cam, 'smplx_kpt_cam': smplx_kpt_cam, 'smplx_pose': smplx_pose, 'smplx_shape': smplx_shape, 'smplx_expr': smplx_expr, 'rmano_root_pose': rmano_root_pose, 'lmano_root_pose': lmano_root_pose}
            meta_info = {'kpt_valid': smplx_kpt_valid, 'kpt_trunc': smplx_kpt_trunc, 'smplx_kpt_valid': smplx_kpt_valid, 'smplx_kpt_trunc': smplx_kpt_trunc, 'smplx_pose_valid': smplx_pose_valid, 'smplx_shape_valid': smplx_shape_valid, 'smplx_expr_valid': smplx_expr_valid, 'rmano_root_valid': rmano_root_valid, 'lmano_root_valid': lmano_root_valid, 'cam_R': cam_param['R'], 'cam_R_valid': float(True), 'is_3D': float(True), 'is_kpt_ih': float(True), 'is_smplx_ih': float(True), 'is_hand_only': float(True)}
        else:
            # mano coordinates (right hand)
            with open(rhand_mano_param_path) as f:
                mano_param = json.load(f)
            mano_param['hand_type'] = 'right'
            _, _, _, rmano_kpt_cam, _, rmano_vert_cam = process_mano_param(mano_param, cam_param, do_flip, img_shape, rot)

            # mano coordinates (left hand)
            with open(lhand_mano_param_path) as f:
                mano_param = json.load(f)
            mano_param['hand_type'] = 'left'
            _, _, _, lmano_kpt_cam, _, lmano_vert_cam = process_mano_param(mano_param, cam_param, do_flip, img_shape, rot)

            inputs = {'img': img}
            targets = {'rmano_kpt_cam': rmano_kpt_cam, 'rmano_vert_cam': rmano_vert_cam, 'lmano_kpt_cam': lmano_kpt_cam, 'lmano_vert_cam': lmano_vert_cam}
            meta_info = {'bb2img_trans': bb2img_trans}

        return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        eval_result = {'mpvpe': [], 'rrve': [], 'mrrpe': []}

        for n in range(sample_num):
            out = outs[n]
            
            # gt
            rhand_vert_gt = out['rmano_vert_cam_target'] * 1000 # meter to millimeter
            lhand_vert_gt = out['lmano_vert_cam_target'] * 1000 # meter to millimeter
            rhand_kpt_gt = np.dot(mano.kpt['regressor'], rhand_vert_gt)
            lhand_kpt_gt = np.dot(mano.kpt['regressor'], lhand_vert_gt)
            rwrist_gt = rhand_kpt_gt[mano.kpt['root_idx']]
            lwrist_gt = lhand_kpt_gt[mano.kpt['root_idx']]
           
            # out
            rhand_vert_out = out['smplx_vert_cam'][smpl_x.hand_vertex_idx['right_hand'],:] * 1000 # meter to millimeter
            lhand_vert_out = out['smplx_vert_cam'][smpl_x.hand_vertex_idx['left_hand'],:] * 1000 # meter to millimeter
            rhand_kpt_out = np.dot(mano.kpt['regressor'], rhand_vert_out)
            lhand_kpt_out = np.dot(mano.kpt['regressor'], lhand_vert_out)
            rwrist_out = rhand_kpt_out[mano.kpt['root_idx']]
            lwrist_out = lhand_kpt_out[mano.kpt['root_idx']]

            # mrrpe
            rel_trans_gt = lwrist_gt - rwrist_gt
            rel_trans_out = lwrist_out - rwrist_out
            eval_result['mrrpe'].append(np.sqrt(np.sum((rel_trans_gt - rel_trans_out)**2)))

            # translation alignment
            rhand_vert_gt = rhand_vert_gt - rwrist_gt
            lhand_vert_gt = lhand_vert_gt - lwrist_gt
            rhand_kpt_gt = rhand_kpt_gt - rwrist_gt
            lhand_kpt_gt = lhand_kpt_gt - lwrist_gt
            rhand_vert_out = rhand_vert_out - rwrist_out
            lhand_vert_out = lhand_vert_out - lwrist_out
            rhand_kpt_out = rhand_kpt_out - rwrist_out
            lhand_kpt_out = lhand_kpt_out - lwrist_out

            # mpvpe
            eval_result['mpvpe'].append(np.sqrt(np.sum((rhand_vert_out - rhand_vert_gt)**2,1)).mean())
            eval_result['mpvpe'].append(np.sqrt(np.sum((lhand_vert_out - lhand_vert_gt)**2,1)).mean())

            # rrve
            lhand_vert_out_trans = lhand_vert_out + rel_trans_out[None,:]
            lhand_vert_gt_trans = lhand_vert_gt + rel_trans_gt[None,:]
            mano_vert_out_trans = np.concatenate((rhand_vert_out, lhand_vert_out_trans))
            mano_vert_gt_trans = np.concatenate((rhand_vert_gt, lhand_vert_gt_trans))
            eval_result['rrve'].append(np.sqrt(np.sum((mano_vert_out_trans - mano_vert_gt_trans)**2,1)).mean())

        return eval_result
    
    def print_eval_result(self, eval_result):
        print('MPVPE: %.2f mm' % (np.mean(eval_result['mpvpe'])))
        print('RRVE: %.2f mm' % (np.mean(eval_result['rrve'])))
        print('MRRPE: %.2f mm' % (np.mean(eval_result['mrrpe'])))
