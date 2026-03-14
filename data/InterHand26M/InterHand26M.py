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
from utils.transforms import world2cam, cam2img, change_kpt_name
from utils.vis import vis_kpt, render_mesh

class InterHand26M(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.img_path = osp.join('..', 'data', 'InterHand26M', 'images')
        self.annot_path = osp.join('..', 'data', 'InterHand26M', 'annotations')
        self.kpt = {
                    'num': 42, 
                    'name': ('R_Thumb_4', 'R_Thumb_3', 'R_Thumb_2', 'R_Thumb_1', 'R_Index_4', 'R_Index_3', 'R_Index_2', 'R_Index_1', 'R_Middle_4', 'R_Middle_3', 'R_Middle_2', 'R_Middle_1', 'R_Ring_4', 'R_Ring_3', 'R_Ring_2', 'R_Ring_1', 'R_Pinky_4', 'R_Pinky_3', 'R_Pinky_2', 'R_Pinky_1', 'R_Wrist', 'L_Thumb_4', 'L_Thumb_3', 'L_Thumb_2', 'L_Thumb_1', 'L_Index_4', 'L_Index_3', 'L_Index_2', 'L_Index_1', 'L_Middle_4', 'L_Middle_3', 'L_Middle_2', 'L_Middle_1', 'L_Ring_4', 'L_Ring_3', 'L_Ring_2', 'L_Ring_1', 'L_Pinky_4', 'L_Pinky_3', 'L_Pinky_2', 'L_Pinky_1', 'L_Wrist'),
                    'flip_pair': [ (i,i+21) for i in range(21)]
                    }
        self.kpt['part_idx'] = {'right': np.arange(0,self.kpt['num']//2), 'left': np.arange(self.kpt['num']//2,self.kpt['num'])}
        self.kpt['root_idx'] = {'right': self.kpt['name'].index('R_Wrist'), 'left': self.kpt['name'].index('L_Wrist')}
        self.datalist = self.load_data()
        
    def load_data(self):
        # load annotation
        db = COCO(osp.join(self.annot_path, self.data_split, 'InterHand2.6M_' + self.data_split + '_data.json'))
        with open(osp.join(self.annot_path, self.data_split, 'InterHand2.6M_' + self.data_split + '_camera.json')) as f:
            cameras = json.load(f)
        with open(osp.join(self.annot_path, self.data_split, 'InterHand2.6M_' + self.data_split + '_joint_3d.json')) as f:
            joints = json.load(f)
        with open(osp.join(self.annot_path, self.data_split, 'InterHand2.6M_' + self.data_split + '_MANO_NeuralAnnot.json')) as f:
            mano_params = json.load(f)
        with open(osp.join(self.annot_path, 'aid_ih_' + self.data_split + '.txt')) as f:
            aid_list = [int(x) for x in f.readlines()]

        datalist = []
        for aid in aid_list:
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_width, img_height = img['width'], img['height']
            img_path = osp.join(self.img_path, self.data_split, img['file_name'])

            capture_id = img['capture']
            seq_name = img['seq_name']
            cam = img['camera']
            frame_idx = img['frame_idx']
            hand_type = ann['hand_type']
            
            # camera parameters
            t, R = np.array(cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32).reshape(3), np.array(cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32).reshape(3,3)
            t = -np.dot(R,t.reshape(3,1)).reshape(3) # -Rt -> t
            focal, princpt = np.array(cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32).reshape(2), np.array(cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32).reshape(2)
            cam_param = {'R': R, 't': t, 'focal': focal, 'princpt': princpt}

            # keypoint coordinates
            kpt_trunc = np.array(ann['joint_valid'], dtype=np.float32).reshape(-1,1)
            kpt_valid = np.array(joints[str(capture_id)][str(frame_idx)]['joint_valid'], dtype=np.float32).reshape(-1,1)
            kpt_world = np.array(joints[str(capture_id)][str(frame_idx)]['world_coord'], dtype=np.float32).reshape(-1,3)
            kpt_cam = world2cam(kpt_world, R, t)
            kpt_cam[np.tile(kpt_valid==0, (1,3))] = 1. # prevent zero division error
            kpt_img = cam2img(kpt_cam, focal, princpt)[:,:2]

            # bbox
            bbox = np.array([0, 0, img_width, img_height], dtype=np.float32)
            bbox = set_aspect_ratio(bbox, cfg.input_img_shape[1]/cfg.input_img_shape[0]) 
            
            # mano parameters
            mano_param = mano_params[str(capture_id)][str(frame_idx)].copy()

            datalist.append({
                'img_path': img_path,
                'img_shape': (img_height, img_width),
                'bbox': bbox,
                'kpt_img': kpt_img,
                'kpt_cam': kpt_cam,
                'kpt_valid': kpt_valid,
                'cam_param': cam_param,
                'mano_param': mano_param,
                'hand_type': hand_type})

        return datalist
    
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox = data['img_path'], data['img_shape'], data['bbox']
        data['cam_param']['t'] /= 1000 # millimeter to meter
        
        # img
        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
        img = self.transform(img.astype(np.float32))/255.
        
        if self.data_split == 'train':
            # mano coordinates (right hand)
            mano_param = data['mano_param']
            mano_param['right']['hand_type'] = 'right'
            rmano_root_pose, rmano_pose, rmano_shape_param, rmano_kpt_cam, rmano_kpt_img, _ = process_mano_param(mano_param['right'], data['cam_param'], do_flip, img_shape, rot)
            rmano_root_valid = float(True)
            rmano_pose_valid = np.ones((mano.joint['num'],1), dtype=np.float32)
            rmano_shape_valid = float(True)
            rmano_kpt_valid = np.ones((mano.kpt['num'],1), dtype=np.float32)

            # mano coordinates (left hand)
            mano_param['left']['hand_type'] = 'left'
            lmano_root_pose, lmano_pose, lmano_shape_param, lmano_kpt_cam, lmano_kpt_img, _ = process_mano_param(mano_param['left'], data['cam_param'], do_flip, img_shape, rot)
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
            is_smplx_ih = float(True)
           
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
            cv2.imwrite('ih26m_' + str(idx) + '_mano.jpg', _img)
            print('saved')
            """
            
            inputs = {'img': img}
            targets = {'kpt_img': smplx_kpt_img, 'smplx_kpt_img': smplx_kpt_img, 'kpt_cam': smplx_kpt_cam, 'smplx_kpt_cam': smplx_kpt_cam, 'smplx_pose': smplx_pose, 'smplx_shape': smplx_shape, 'smplx_expr': smplx_expr, 'rmano_root_pose': rmano_root_pose, 'lmano_root_pose': lmano_root_pose}
            meta_info = {'kpt_valid': smplx_kpt_valid, 'kpt_trunc': smplx_kpt_trunc, 'smplx_kpt_valid': smplx_kpt_valid, 'smplx_kpt_trunc': smplx_kpt_trunc, 'smplx_pose_valid': smplx_pose_valid, 'smplx_shape_valid': smplx_shape_valid, 'smplx_expr_valid': smplx_expr_valid, 'rmano_root_valid': rmano_root_valid, 'lmano_root_valid': lmano_root_valid, 'cam_R': data['cam_param']['R'], 'cam_R_valid': float(True), 'is_3D': float(True), 'is_kpt_ih': is_smplx_ih, 'is_smplx_ih': is_smplx_ih, 'is_hand_only': float(True)}
        else:
            # mano coordinates (right hand)
            mano_param = data['mano_param']
            mano_param['right']['hand_type'] = 'right'
            _, _, _, _, _, rmano_vert_cam_wo_aug = process_mano_param(mano_param['right'], data['cam_param'], do_flip, img_shape, rot)

            # mano coordinates (left hand)
            mano_param['left']['hand_type'] = 'left'
            _, _, _, _, _, lmano_vert_cam_wo_aug = process_mano_param(mano_param['left'], data['cam_param'], do_flip, img_shape, rot)
           
            inputs = {'img': img}
            targets = {'rmano_vert_cam': rmano_vert_cam_wo_aug, 'lmano_vert_cam': lmano_vert_cam_wo_aug}
            meta_info = {'bb2img_trans': bb2img_trans}

        return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        eval_result = {'mpjpe': [], 'mpvpe': [], 'mrrpe': []}
        for n in range(sample_num):
            out = outs[n]
 
            # gt
            rhand_vert_gt = out['rmano_vert_cam_target'] * 1000 # meter to millimeter
            lhand_vert_gt = out['lmano_vert_cam_target'] * 1000 # meter to millimeter
            rhand_kpt_gt = np.dot(mano.kpt['regressor'], rhand_vert_gt)
            lhand_kpt_gt = np.dot(mano.kpt['regressor'], lhand_vert_gt)
            rwrist_gt = rhand_kpt_gt[mano.kpt['root_idx']]
            lwrist_gt = lhand_kpt_gt[mano.kpt['root_idx']]
            rmiddle_gt = rhand_kpt_gt[mano.kpt['name'].index('Middle_1')]
            lmiddle_gt = lhand_kpt_gt[mano.kpt['name'].index('Middle_1')]
            rlength_gt = np.sqrt(np.sum((rwrist_gt - rmiddle_gt)**2))
            llength_gt = np.sqrt(np.sum((lwrist_gt - lmiddle_gt)**2))
            
            # out
            rhand_vert_out = out['smplx_vert_cam'][smpl_x.hand_vertex_idx['right_hand'],:] * 1000 # meter to millimeter
            lhand_vert_out = out['smplx_vert_cam'][smpl_x.hand_vertex_idx['left_hand'],:] * 1000 # meter to millimeter
            rhand_kpt_out = np.dot(mano.kpt['regressor'], rhand_vert_out)
            lhand_kpt_out = np.dot(mano.kpt['regressor'], lhand_vert_out)
            rwrist_out = rhand_kpt_out[mano.kpt['root_idx']]
            lwrist_out = lhand_kpt_out[mano.kpt['root_idx']]
            rmiddle_out = rhand_kpt_out[mano.kpt['name'].index('Middle_1')]
            lmiddle_out = lhand_kpt_out[mano.kpt['name'].index('Middle_1')]
            rlength_out = np.sqrt(np.sum((rwrist_out - rmiddle_out)**2))
            llength_out = np.sqrt(np.sum((lwrist_out - lmiddle_out)**2))
         
            # mrrpe
            rel_trans_gt = lwrist_gt - rwrist_gt
            rel_trans_out = lwrist_out - rwrist_out
            eval_result['mrrpe'].append(np.sqrt(np.sum((rel_trans_gt - rel_trans_out)**2)))

            # translation alignment (use Middle_1 for the alignment following previous works (e.g., IntagHand))
            rhand_vert_gt = rhand_vert_gt - rmiddle_gt
            lhand_vert_gt = lhand_vert_gt - lmiddle_gt
            rhand_kpt_gt = rhand_kpt_gt - rmiddle_gt
            lhand_kpt_gt = lhand_kpt_gt - lmiddle_gt
            rhand_vert_out = rhand_vert_out - rmiddle_out
            lhand_vert_out = lhand_vert_out - lmiddle_out
            rhand_kpt_out = rhand_kpt_out - rmiddle_out
            lhand_kpt_out = lhand_kpt_out - lmiddle_out

            # scale alignment
            rhand_scale = rlength_gt / rlength_out
            lhand_scale = llength_gt / llength_out
            rhand_vert_out = rhand_vert_out * rhand_scale
            lhand_vert_out = lhand_vert_out * lhand_scale
            rhand_kpt_out = rhand_kpt_out * rhand_scale
            lhand_kpt_out = lhand_kpt_out * lhand_scale

            # mpjpe
            eval_result['mpjpe'].append(np.sqrt(np.sum((rhand_kpt_out - rhand_kpt_gt)**2,1)).mean())
            eval_result['mpjpe'].append(np.sqrt(np.sum((lhand_kpt_out - lhand_kpt_gt)**2,1)).mean())

            # mpvpe
            eval_result['mpvpe'].append(np.sqrt(np.sum((rhand_vert_out - rhand_vert_gt)**2,1)).mean())
            eval_result['mpvpe'].append(np.sqrt(np.sum((lhand_vert_out - lhand_vert_gt)**2,1)).mean())

        return eval_result
    
    def print_eval_result(self, eval_result):
        print('MPJPE: %.2f mm' % (np.mean(eval_result['mpjpe'])))       
        print('MPVPE: %.2f mm' % (np.mean(eval_result['mpvpe'])))
        print('MRRPE: %.2f mm' % (np.mean(eval_result['mrrpe'])))



