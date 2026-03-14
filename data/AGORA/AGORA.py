import os
import os.path as osp
from glob import glob
import numpy as np
from config import cfg
import copy
import json
import pickle
import cv2
import torch
from pycocotools.coco import COCO
from utils.smpl_x import smpl_x
from utils.mano import mano
from utils.preprocessing import load_img, set_aspect_ratio, augmentation, process_smplx_param
from utils.transforms import rigid_align
from utils.vis import vis_kpt

class AGORA(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.data_path = osp.join('..', 'data', 'AGORA', 'data')
        self.resolution = (2160, 3840) # height, width. one of (720, 1280) and (2160, 3840)
        self.test_set = 'val' # val, test
        self.datalist = self.load_data()
 
    def load_data(self):
        datalist = []
        if self.data_split == 'train' or (self.data_split == 'test' and self.test_set == 'val'): 
            if self.data_split == 'train':
                db = COCO(osp.join(self.data_path, 'AGORA_train_SMPLX.json'))
            else:
                db = COCO(osp.join(self.data_path, 'AGORA_validation_SMPLX.json'))
            
            for aid in db.anns.keys():
                ann = db.anns[aid]
                image_id = ann['image_id']
                img = db.loadImgs(image_id)[0]
                person_id = ann['person_id']
                if not ann['is_valid']:
                    continue
                
                smplx_param_path = osp.join(self.data_path, ann['smplx_param_path'])
                cam_param_path = osp.join(self.data_path, ann['cam_param_path'])
               
                if self.resolution == (720, 1280):
                    img_shape = self.resolution
                    img_path = osp.join(self.data_path, img['file_name_1280x720'])
                    
                    # convert to current resolution
                    bbox = np.array(ann['bbox']).reshape(2,2)
                    bbox[:,0] = bbox[:,0] / 3840 * 1280
                    bbox[:,1] = bbox[:,1] / 2160 * 720
                    bbox = bbox.reshape(4)
                    bbox = set_aspect_ratio(bbox, cfg.input_img_shape[1]/cfg.input_img_shape[0])

                    if self.data_split == 'train':
                        gender = 'neutral'
                    else:
                        gender = ann['gender']

                    data_dict = {'img_path': img_path, 'img_shape': img_shape, 'bbox': bbox, 'smplx_param_path': smplx_param_path, 'cam_param_path': cam_param_path, 'gender': gender}
                    datalist.append(data_dict)

                elif self.resolution == (2160, 3840): # use cropped and resized images. loading 4K images in pytorch dataloader takes too much time...
                    img_path = osp.join(self.data_path, 'images_3840x2160', img['file_name_3840x2160'].split('/')[-2] + '_crop', img['file_name_3840x2160'].split('/')[-1][:-4] + '_person_id_' + str(person_id) + '.png')
                    json_path = osp.join(self.data_path, 'images_3840x2160', img['file_name_3840x2160'].split('/')[-2] + '_crop', img['file_name_3840x2160'].split('/')[-1][:-4] + '_person_id_' + str(person_id) + '.json')
                    
                    with open(json_path) as f:
                        crop_resize_info = json.load(f)
                        img2bb_trans_from_orig = np.array(crop_resize_info['affine_mat'], dtype=np.float32)
                        resized_height, resized_width = crop_resize_info['resized_height'], crop_resize_info['resized_width']
                    img_shape = (resized_height, resized_width)
                    bbox = np.array([0, 0, resized_width, resized_height], dtype=np.float32)

                    if self.data_split == 'train':
                        gender = 'neutral'
                    else:
                        gender = ann['gender']

                    data_dict = {'img_path': img_path, 'img_shape': img_shape, 'bbox': bbox, 'img2bb_trans_from_orig': img2bb_trans_from_orig, 'smplx_param_path': smplx_param_path, 'cam_param_path': cam_param_path, 'gender': gender}
                    datalist.append(data_dict)

        elif self.data_split == 'test' and self.test_set == 'test':
            with open(osp.join(self.data_path, 'AGORA_test_bbox.json')) as f:
                bboxs = json.load(f)

            for filename in bboxs.keys():
                if self.resolution == (720, 1280):
                    img_path = osp.join(self.data_path, 'test', filename)
                    img_shape = self.resolution
                    person_num = len(bboxs[filename])
                    for person_id in range(person_num):
                        # change bbox from (2160,3840) to target resoution
                        bbox = np.array(bboxs[filename][person_id]['bbox']).reshape(2,2)
                        bbox[:,0] = bbox[:,0] / 3840 * 1280
                        bbox[:,1] = bbox[:,1] / 2160 * 720
                        bbox = bbox.reshape(4)
                        bbox = set_aspect_ratio(bbox, cfg.input_img_shape[1]/cfg.input_img_shape[0])
                        if bbox is None:
                            continue
                        datalist.append({'img_path': img_path, 'img_shape': img_shape, 'bbox': bbox, 'person_id': person_id})

                elif self.resolution == (2160, 3840): # use cropped and resized images. loading 4K images in pytorch dataloader takes too much time...
                    person_num = len(bboxs[filename])
                    for person_id in range(person_num):
                        img_path = osp.join(self.data_path, 'images_3840x2160', 'test_crop', filename[:-4] + '_person_id_' + str(person_id) + '.png')
                        json_path = osp.join(self.data_path, 'images_3840x2160', 'test_crop', filename[:-4] + '_person_id_' + str(person_id) + '.json')
                        if not osp.isfile(json_path):
                            continue
                        with open(json_path) as f:
                            crop_resize_info = json.load(f)
                            img2bb_trans_from_orig = np.array(crop_resize_info['affine_mat'], dtype=np.float32)
                            resized_height, resized_width = crop_resize_info['resized_height'], crop_resize_info['resized_width']
                        img_shape = (resized_height, resized_width)
                        bbox = np.array([0, 0, resized_width, resized_height], dtype=np.float32)
                        datalist.append({'img_path': img_path, 'img_shape': img_shape, 'img2bb_trans_from_orig': img2bb_trans_from_orig, 'bbox': bbox, 'person_id': person_id})
        
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
            with open(data['smplx_param_path']) as f:
                smplx_param = json.load(f)
            root_pose = np.array(smplx_param['global_orient'], dtype=np.float32).reshape(-1)
            body_pose = np.array(smplx_param['body_pose'], dtype=np.float32).reshape(-1)
            shape = np.array(smplx_param['betas'], dtype=np.float32).reshape(-1)
            lhand_pose = np.array(smplx_param['left_hand_pose'], dtype=np.float32).reshape(-1)
            rhand_pose = np.array(smplx_param['right_hand_pose'], dtype=np.float32).reshape(-1)
            jaw_pose = np.array(smplx_param['jaw_pose'], dtype=np.float32).reshape(-1)
            leye_pose = np.array(smplx_param['leye_pose'], dtype=np.float32).reshape(-1)
            reye_pose = np.array(smplx_param['reye_pose'], dtype=np.float32).reshape(-1)
            expr = np.array(smplx_param['expression'], dtype=np.float32).reshape(-1)
            trans = np.array(smplx_param['transl'], dtype=np.float32).reshape(-1)
            smplx_param = {'root_pose': root_pose, 'body_pose': body_pose, 'shape': shape,
                    'lhand_pose': lhand_pose, 'lhand_valid': True, 
                    'rhand_pose': rhand_pose, 'rhand_valid': True, 
                    'jaw_pose': jaw_pose, 'leye_pose': leye_pose, 'reye_pose': reye_pose, 'expr': expr, 'face_valid': True,
                    'trans': trans, 'gender': data['gender']}
            if data['gender'] == 'neutral':
                with open(data['smplx_param_path'].replace('.json', '_neutral_gender_betas.json')) as f:
                    smplx_param['shape'] = np.array(json.load(f), dtype=np.float32)
            with open(data['cam_param_path']) as f:
                cam_param = {k: np.array(v, dtype=np.float32) for k,v in json.load(f).items()}
            if self.resolution == (2160, 3840): # apply crop and resize
                cam_param['focal'][0] = cam_param['focal'][0] * data['img2bb_trans_from_orig'][0][0]
                cam_param['focal'][1] = cam_param['focal'][1] * data['img2bb_trans_from_orig'][1][1]
                cam_param['princpt'][0] = cam_param['princpt'][0] * data['img2bb_trans_from_orig'][0][0] + data['img2bb_trans_from_orig'][0][2]
                cam_param['princpt'][1] = cam_param['princpt'][1] * data['img2bb_trans_from_orig'][1][1] + data['img2bb_trans_from_orig'][1][2]
            else: # scale camera parameters
                cam_param['focal'][0] = cam_param['focal'][0] / 3840 * 1280
                cam_param['focal'][1] = cam_param['focal'][1] / 2160 * 720
                cam_param['princpt'][0] = cam_param['princpt'][0] / 3840 * 1280
                cam_param['princpt'][1] = cam_param['princpt'][1] / 2160 * 720
            smplx_pose, smplx_shape, smplx_expr, smplx_pose_valid, smplx_expr_valid, smplx_kpt_cam, smplx_kpt_img, smplx_kpt_valid, smplx_kpt_trunc, _ = process_smplx_param(smplx_param, cam_param, do_flip, img_shape, img2bb_trans, rot)
            smplx_shape_valid = True
            
            """
            # for debug
            _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
            _tmp = smplx_kpt_img.copy()
            _tmp[:,0] = _tmp[:,0] / cfg.vit_output_shape[1] * cfg.input_img_shape[1]
            _tmp[:,1] = _tmp[:,1] / cfg.vit_output_shape[0] * cfg.input_img_shape[0]
            _img = vis_kpt(_img, _tmp)
            cv2.imwrite('agora_' + str(idx) + '.jpg', _img)
            print('saved')
            """

            # dummy mano root pose
            dummy_root_pose = np.zeros((3), dtype=np.float32)

            # dummy camera extrinsic
            dummy_cam_R = np.eye(3).astype(np.float32)

            inputs = {'img': img}
            targets = {'kpt_img': smplx_kpt_img, 'kpt_cam': smplx_kpt_cam, 'smplx_kpt_img': smplx_kpt_img, 'smplx_kpt_cam': smplx_kpt_cam, 'smplx_pose': smplx_pose, 'smplx_shape': smplx_shape, 'smplx_expr': smplx_expr, 'rmano_root_pose': dummy_root_pose, 'lmano_root_pose': dummy_root_pose}
            meta_info = {'kpt_valid': smplx_kpt_valid, 'kpt_trunc': smplx_kpt_trunc, 'smplx_kpt_valid': smplx_kpt_valid, 'smplx_kpt_trunc': smplx_kpt_trunc, 'smplx_pose_valid': smplx_pose_valid, 'smplx_shape_valid': float(smplx_shape_valid), 'smplx_expr_valid': float(smplx_expr_valid), 'rmano_root_valid': float(False), 'lmano_root_valid': float(False), 'cam_R': dummy_cam_R, 'cam_R_valid': float(False), 'is_3D': float(True), 'is_kpt_ih': float(False), 'is_smplx_ih': float(False), 'is_hand_only': float(False)}
            return inputs, targets, meta_info
        else:
            # load crop and resize information (for the 4K setting)
            if self.resolution == (2160, 3840):
                img2bb_trans = np.dot(
                                    np.concatenate((img2bb_trans,
                                                    np.array([0,0,1], dtype=np.float32).reshape(1,3))),
                                    np.concatenate((data['img2bb_trans_from_orig'],
                                                    np.array([0,0,1], dtype=np.float32).reshape(1,3)))
                                    )
                bb2img_trans = np.linalg.inv(img2bb_trans)[:2,:]
                img2bb_trans = img2bb_trans[:2,:]

            if self.test_set == 'val':
                # smplx parameters
                with open(data['smplx_param_path']) as f:
                    smplx_param = json.load(f)
                root_pose = np.array(smplx_param['global_orient'], dtype=np.float32).reshape(-1)
                body_pose = np.array(smplx_param['body_pose'], dtype=np.float32).reshape(-1)
                shape = np.array(smplx_param['betas'], dtype=np.float32).reshape(-1)[:10]
                lhand_pose = np.array(smplx_param['left_hand_pose'], dtype=np.float32).reshape(-1)
                rhand_pose = np.array(smplx_param['right_hand_pose'], dtype=np.float32).reshape(-1)
                jaw_pose = np.array(smplx_param['jaw_pose'], dtype=np.float32).reshape(-1)
                leye_pose = np.array(smplx_param['leye_pose'], dtype=np.float32).reshape(-1)
                reye_pose = np.array(smplx_param['reye_pose'], dtype=np.float32).reshape(-1)
                expr = np.array(smplx_param['expression'], dtype=np.float32).reshape(-1)
                trans = np.array(smplx_param['transl'], dtype=np.float32).reshape(-1)
                with open(data['cam_param_path']) as f:
                    cam_param = json.load(f)
                smplx_param = {'root_pose': root_pose, 'body_pose': body_pose, 'shape': shape,
                        'lhand_pose': lhand_pose, 'lhand_valid': True,
                        'rhand_pose': rhand_pose, 'rhand_valid': True,
                        'jaw_pose': jaw_pose, 'leye_pose': leye_pose, 'reye_pose': reye_pose, 'expr': expr, 'face_valid': True,
                        'trans': trans, 'gender': data['gender']}
                if data['gender'] == 'neutral':
                    with open(data['smplx_param_path'].replace('.json', '_neutral_gender_betas.json')) as f:
                        smplx_param['shape'] = np.array(json.load(f), dtype=np.float32)
                _, _, _, _, _, _, _, _, _, smplx_vert_cam_wo_aug = process_smplx_param(smplx_param, cam_param, do_flip, img_shape, img2bb_trans, rot)

                inputs = {'img': img}
                targets = {'smplx_vert_cam': smplx_vert_cam_wo_aug}
                meta_info = {'bb2img_trans': bb2img_trans}
            else:
                inputs = {'img': img}
                targets = {'smplx_vert_cam': np.zeros((smpl_x.vertex_num, 3), dtype=np.float32)} # dummy vertex
                meta_info = {'bb2img_trans': bb2img_trans}

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
            vert_out_align = rigid_align(vert_out, vert_gt)
            eval_result['pa_mpvpe_all'].append(np.sqrt(np.sum((vert_out_align - vert_gt)**2,1)).mean() * 1000)
            vert_out_align = vert_out - np.dot(smpl_x.vert_to_joint, vert_out)[smpl_x.vert_to_joint_idx['pelvis'],None,:] + np.dot(smpl_x.vert_to_joint, vert_gt)[smpl_x.vert_to_joint_idx['pelvis'],None,:]
            eval_result['mpvpe_all'].append(np.sqrt(np.sum((vert_out_align - vert_gt)**2,1)).mean() * 1000)

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

