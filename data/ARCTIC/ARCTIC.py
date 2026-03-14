import os
import os.path as osp
from glob import glob
import numpy as np
from config import cfg
import copy
import json
import cv2
import torch
import random
from utils.smpl_x import smpl_x
from utils.mano import mano
from utils.preprocessing import load_img, get_bbox, set_aspect_ratio, augmentation, process_smplx_param
from utils.transforms import rigid_align
from utils.vis import vis_kpt

class ARCTIC(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split 
        self.test_set = 'val' # 'val', 'test'
        self.data_path = osp.join('..', 'data', 'ARCTIC', 'data')
        if data_split == 'train':
            self.frame_downsample_factor = 6
        else:
            self.frame_downsample_factor = 10
        self.img_downsample_factor = 2
        self.datalist = self.load_data()
 
    def load_data(self):
        with open(osp.join(self.data_path, 'meta', 'misc.json')) as f:
            db_info = json.load(f)
        with open(osp.join(self.data_path, 'splits_json', 'protocol_p1.json')) as f:
            if self.data_split == 'train':
                split_info = json.load(f)['train']
            else:
                split_info = json.load(f)[self.test_set]

        datalist = []
        for subject_name_seq_name in split_info:
            subject_name, seq_name = subject_name_seq_name.split('/')

            if (self.data_split == 'train') or ((self.data_split == 'test') and (self.test_set == 'val')):
                ego_cam_param = np.load(osp.join(self.data_path, 'raw_seqs', subject_name, seq_name + '.egocam.dist.npy'), allow_pickle=True)[()]
                smplx_params = np.load(osp.join(self.data_path, 'raw_seqs', subject_name, seq_name + '.smplx.npy'), allow_pickle=True)[()]
                if self.data_split == 'train':
                    with open(osp.join(self.data_path, 'smplx_shape_param', 'neutral_gender', subject_name + '_smplx_shape_param.json')) as f:
                        smplx_shape_param = np.array(json.load(f), dtype=np.float32)
                else:
                    with open(osp.join(self.data_path, 'smplx_shape_param', 'gendered', subject_name + '_smplx_shape_param.json')) as f:
                        smplx_shape_param = np.array(json.load(f), dtype=np.float32)
                    #with open(osp.join(self.data_path, 'smplx_shape_param', 'neutral_gender', subject_name + '_smplx_shape_param.json')) as f:
                    #    smplx_shape_param = np.array(json.load(f), dtype=np.float32)
                with open(osp.join(self.data_path, 'bbox', subject_name, seq_name + '.json')) as f:
                    bbox = json.load(f)

                cam_path_list = [x for x in glob(osp.join(self.data_path, 'images_1k', subject_name, seq_name, '*')) if x.split('/')[-1] != 'data']
                for cam_path in cam_path_list:
                    cam_name = cam_path.split('/')[-1]
                    if cam_name == '0':
                        is_ego = True
                    else:
                        is_ego = False
                    if self.data_split == 'train':
                        gender = 'neutral'
                    else:
                        gender = 'neutral' 
                        #gender = db_info[subject_name]['gender']
                    img_shape = (db_info[subject_name]['image_size'][int(cam_name)][1]//self.img_downsample_factor, \
                                db_info[subject_name]['image_size'][int(cam_name)][0]//self.img_downsample_factor) # height, width

                    img_path_list = glob(osp.join(cam_path, '*.jpg'))
                    for img_path in img_path_list:
                        frame_idx = int(img_path.split('/')[-1][:-4])
                        # sample frames
                        if frame_idx % self.frame_downsample_factor != 0:
                            continue

                        if str(frame_idx) not in bbox[cam_name]:
                            continue
                        if bbox[cam_name][str(frame_idx)]['body_bbox'] is None:
                                continue

                        # bbox
                        body_bbox = set_aspect_ratio(np.array(bbox[cam_name][str(frame_idx)]['body_bbox'], dtype=np.float32)/self.img_downsample_factor, cfg.input_img_shape[1]/cfg.input_img_shape[0])
                        rhand_bbox, lhand_bbox, face_bbox = bbox[cam_name][str(frame_idx)]['rhand_bbox'], bbox[cam_name][str(frame_idx)]['lhand_bbox'], bbox[cam_name][str(frame_idx)]['face_bbox']
                        if (rhand_bbox is None) and (lhand_bbox is None):
                            continue

                        offset = db_info[subject_name]['ioi_offset']
                        frame_idx_offset = frame_idx - offset
                        if (frame_idx_offset < 0) or (frame_idx_offset >= smplx_params['global_orient'].shape[0]):
                            continue

                        # camera parameter
                        if is_ego:
                            cam_param = {'R': ego_cam_param['R_k_cam_np'][frame_idx_offset], \
                                        't': ego_cam_param['T_k_cam_np'][frame_idx_offset], \
                                        'focal': np.array([ego_cam_param['intrinsics'][0][0], ego_cam_param['intrinsics'][1][1]], dtype=np.float32)/self.img_downsample_factor, \
                                        'princpt': np.array([ego_cam_param['intrinsics'][0][2], ego_cam_param['intrinsics'][1][2]], dtype=np.float32)/self.img_downsample_factor}
                        else:
                            extrinsic = np.array(db_info[subject_name]['world2cam'], dtype=np.float32)[int(cam_name)-1]
                            intrinsic = np.array(db_info[subject_name]['intris_mat'], dtype=np.float32)[int(cam_name)-1]
                            cam_param = {'R': extrinsic[:3,:3], \
                                        't': extrinsic[:3,3], \
                                        'focal': np.array([intrinsic[0][0], intrinsic[1][1]], dtype=np.float32)/self.img_downsample_factor, \
                                        'princpt': np.array([intrinsic[0][2], intrinsic[1][2]], dtype=np.float32)/self.img_downsample_factor}
                        
                        # smplx parameter
                        smplx_param = {
                                    'root_pose': smplx_params['global_orient'][frame_idx_offset],
                                    'body_pose': smplx_params['body_pose'][frame_idx_offset],
                                    'jaw_pose': smplx_params['jaw_pose'][frame_idx_offset],
                                    'leye_pose': smplx_params['leye_pose'][frame_idx_offset],
                                    'reye_pose': smplx_params['reye_pose'][frame_idx_offset],
                                    'lhand_pose': smplx_params['left_hand_pose'][frame_idx_offset],
                                    'rhand_pose': smplx_params['right_hand_pose'][frame_idx_offset],
                                    'trans': smplx_params['transl'][frame_idx_offset],
                                    'shape': smplx_shape_param,
                                    'lhand_valid': True,
                                    'rhand_valid': True,
                                    'face_valid': True,
                                    'gender': gender
                                    }
                        smplx_param['lhand_pose'] = smplx_param['lhand_pose'] - smpl_x.layer['neutral'].left_hand_mean.numpy().reshape(-1) # flat_hand_mean=True -> flat_hand_mean=False
                        smplx_param['rhand_pose'] = smplx_param['rhand_pose'] - smpl_x.layer['neutral'].right_hand_mean.numpy().reshape(-1) # flat_hand_mean=True -> flat_hand_mean=False

                        data_dict = {'img_path': img_path, 'img_shape': img_shape, 'bbox': body_bbox, 'rhand_bbox': rhand_bbox, 'lhand_bbox': lhand_bbox, 'cam_param': cam_param, 'smplx_param': smplx_param, 'is_ego': is_ego}
                        datalist.append(data_dict)

            else:
                with open(osp.join(self.data_path, 'bbox', subject_name, seq_name + '.json')) as f:
                    bbox = json.load(f)

                cam_path_list = [x for x in glob(osp.join(self.data_path, 'images_1k', subject_name, seq_name, '*')) if x.split('/')[-1] != 'data']
                for cam_path in cam_path_list:
                    cam_name = cam_path.split('/')[-1]
                    if cam_name == '0':
                        is_ego = True
                    else:
                        is_ego = False
                    gender = db_info[subject_name]['gender']
                    img_shape = (db_info[subject_name]['image_size'][int(cam_name)][1]//self.img_downsample_factor, \
                                db_info[subject_name]['image_size'][int(cam_name)][0]//self.img_downsample_factor) # height, width

                    img_path_list = glob(osp.join(cam_path, '*.jpg'))
                    for img_path in img_path_list:
                        frame_idx = int(img_path.split('/')[-1][:-4])
                        if str(frame_idx) not in bbox[cam_name]:
                            continue
                        if bbox[cam_name][str(frame_idx)] is None:
                                continue

                        # bbox
                        body_bbox = set_aspect_ratio(np.array(bbox[cam_name][str(frame_idx)], dtype=np.float32)/self.img_downsample_factor, cfg.input_img_shape[1]/cfg.input_img_shape[0])
                        offset = db_info[subject_name]['ioi_offset']
                        frame_idx_offset = frame_idx - offset

                        data_dict = {'img_path': img_path, 'img_shape': img_shape, 'bbox': body_bbox}
                        datalist.append(data_dict)

        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox = data['img_path'], data['img_shape'], data['bbox']

        # image load
        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
        img = self.transform(img.astype(np.float32))/255.
        
        if self.data_split == 'train':
            # smplx parameters
            smplx_pose, smplx_shape, smplx_expr, smplx_pose_valid, smplx_expr_valid, smplx_kpt_cam, smplx_kpt_img, smplx_kpt_valid, smplx_kpt_trunc, _ = process_smplx_param(data['smplx_param'], data['cam_param'], do_flip, img_shape, img2bb_trans, rot)
            smplx_shape_valid = True
            rhand, lhand = smplx_kpt_cam[smpl_x.kpt['part_idx']['rhand'],:], smplx_kpt_cam[smpl_x.kpt['part_idx']['lhand'],:]
            if np.sqrt(np.sum((rhand.mean(0) - lhand.mean(0))**2)) < 0.2:
                is_smplx_ih = float(True)
            else:
                is_smplx_ih = float(False)

            """
            # for debug
            _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
            _tmp = smplx_kpt_img.copy()
            _tmp[:,0] = _tmp[:,0] / cfg.vit_output_shape[1] * cfg.input_img_shape[1]
            _tmp[:,1] = _tmp[:,1] / cfg.vit_output_shape[0] * cfg.input_img_shape[0]
            _img = vis_kpt(_img, _tmp)
            cv2.imwrite('ARCTIC_' + str(idx) + '.jpg', _img)
            print('saved')
            """
            
            is_ego = data['is_ego']
            if is_ego:
                is_hand_only = float(True)
            else:
                is_hand_only = float(False)

            # fill with dummy MANO values
            dummy_root_pose = np.zeros((3), dtype=np.float32)

            inputs = {'img': img}
            targets = {'kpt_img': smplx_kpt_img, 'kpt_cam': smplx_kpt_cam, 'smplx_kpt_img': smplx_kpt_img, 'smplx_kpt_cam': smplx_kpt_cam, 'smplx_pose': smplx_pose, 'smplx_shape': smplx_shape, 'smplx_expr': smplx_expr, 'rmano_root_pose': dummy_root_pose, 'lmano_root_pose': dummy_root_pose}
            meta_info = {'kpt_valid': smplx_kpt_valid, 'kpt_trunc': smplx_kpt_trunc, 'smplx_kpt_valid': smplx_kpt_valid, 'smplx_kpt_trunc': smplx_kpt_trunc, 'smplx_pose_valid': smplx_pose_valid, 'smplx_shape_valid': float(smplx_shape_valid), 'smplx_expr_valid': float(smplx_expr_valid), 'rmano_root_valid': float(False), 'lmano_root_valid': float(False), 'cam_R': data['cam_param']['R'], 'cam_R_valid': float(True), 'is_3D': float(True), 'is_kpt_ih': is_smplx_ih, 'is_smplx_ih': is_smplx_ih, 'is_hand_only': is_hand_only}
            return inputs, targets, meta_info

        else:
            if self.test_set == 'val':
                _, _, _, _, _, _, _, _, _, smplx_vert_cam_wo_aug = process_smplx_param(data['smplx_param'], data['cam_param'], do_flip, img_shape, img2bb_trans, rot)

                inputs = {'img': img}
                targets = {'smplx_vert_cam': smplx_vert_cam_wo_aug}
                meta_info = {'bb2img_trans': bb2img_trans}
                return inputs, targets, meta_info
            else:
                inputs = {'img': img}
                targets = {}
                meta_info = {}
                return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        eval_result = {'pa_mpvpe_all': [], 'pa_mpvpe_hand': [], 'pa_mpvpe_face': [], 'mpvpe_all': [], 'mpvpe_hand': [], 'mpvpe_face': []}
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            out = outs[n]
            vert_gt = out['smplx_vert_cam_target']
            vert_out = out['smplx_vert_cam']
           
            # MPVPE from all vertices
            vert_out_align = vert_out - np.dot(smpl_x.vert_to_joint, vert_out)[smpl_x.vert_to_joint_idx['pelvis'],None,:] + np.dot(smpl_x.vert_to_joint, vert_gt)[smpl_x.vert_to_joint_idx['pelvis'],None,:]
            eval_result['mpvpe_all'].append(np.sqrt(np.sum((vert_out_align - vert_gt)**2,1)).mean() * 1000)
            vert_out_align = rigid_align(vert_out, vert_gt)
            eval_result['pa_mpvpe_all'].append(np.sqrt(np.sum((vert_out_align - vert_gt)**2,1)).mean() * 1000)

            # MPVPE from hand vertices
            vert_gt_lhand = vert_gt[smpl_x.hand_vertex_idx['left_hand'],:]
            vert_out_lhand = vert_out[smpl_x.hand_vertex_idx['left_hand'],:]
            vert_gt_rhand = vert_gt[smpl_x.hand_vertex_idx['right_hand'],:]
            vert_out_rhand = vert_out[smpl_x.hand_vertex_idx['right_hand'],:]
            vert_out_lhand_align = vert_out_lhand - np.dot(smpl_x.vert_to_joint, vert_out)[smpl_x.vert_to_joint_idx['lwrist'],None,:] + np.dot(smpl_x.vert_to_joint, vert_gt)[smpl_x.vert_to_joint_idx['lwrist'],None,:]
            vert_out_rhand_align = vert_out_rhand - np.dot(smpl_x.vert_to_joint, vert_out)[smpl_x.vert_to_joint_idx['rwrist'],None,:] + np.dot(smpl_x.vert_to_joint, vert_gt)[smpl_x.vert_to_joint_idx['rwrist'],None,:]
            eval_result['mpvpe_hand'].append((np.sqrt(np.sum((vert_out_lhand_align - vert_gt_lhand)**2,1)).mean() * 1000 + np.sqrt(np.sum((vert_out_rhand_align - vert_gt_rhand)**2,1)).mean() * 1000)/2.)
            vert_out_lhand_align = rigid_align(vert_out_lhand, vert_gt_lhand)
            vert_out_rhand_align = rigid_align(vert_out_rhand, vert_gt_rhand)
            eval_result['pa_mpvpe_hand'].append((np.sqrt(np.sum((vert_out_lhand_align - vert_gt_lhand)**2,1)).mean() * 1000 + np.sqrt(np.sum((vert_out_rhand_align - vert_gt_rhand)**2,1)).mean() * 1000)/2.)

            # MPVPE from face vertices
            vert_gt_face = vert_gt[smpl_x.face_vertex_idx,:]
            vert_out_face = vert_out[smpl_x.face_vertex_idx,:]
            vert_out_face_align = vert_out_face - np.dot(smpl_x.vert_to_joint, vert_out)[smpl_x.vert_to_joint_idx['neck'],None,:] + np.dot(smpl_x.vert_to_joint, vert_gt)[smpl_x.vert_to_joint_idx['neck'],None,:]
            eval_result['mpvpe_face'].append(np.sqrt(np.sum((vert_out_face_align - vert_gt_face)**2,1)).mean() * 1000)
            vert_out_face_align = rigid_align(vert_out_face, vert_gt_face)
            eval_result['pa_mpvpe_face'].append(np.sqrt(np.sum((vert_out_face_align - vert_gt_face)**2,1)).mean() * 1000)
            
        return eval_result

    def print_eval_result(self, eval_result):
        print('PA MPVPE (All): %.2f mm' % np.mean(eval_result['pa_mpvpe_all']))
        print('PA MPVPE (Hands): %.2f mm' % np.mean(eval_result['pa_mpvpe_hand']))
        print('PA MPVPE (Face): %.2f mm' % np.mean(eval_result['pa_mpvpe_face']))

        print('MPVPE (All): %.2f mm' % np.mean(eval_result['mpvpe_all']))
        print('MPVPE (Hands): %.2f mm' % np.mean(eval_result['mpvpe_hand']))
        print('MPVPE (Face): %.2f mm' % np.mean(eval_result['mpvpe_face']))


