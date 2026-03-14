import os
import os.path as osp
from glob import glob
import numpy as np
from config import cfg
import copy
import json
import cv2
import torch
from pycocotools.coco import COCO
from utils.smpl_x import smpl_x
from utils.preprocessing import load_img, set_aspect_ratio, augmentation
from utils.transforms import rigid_align
from pytorch3d.io import load_ply

class EHF(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.data_path = osp.join('..', 'data', 'EHF', 'data')
        self.datalist = self.load_data()
        self.cam_param = {'R': [-2.98747896, 0.01172457, -0.05704687]}
        self.cam_param['R'], _ = cv2.Rodrigues(np.array(self.cam_param['R']))
 
    def load_data(self):
        datalist = []
        db = COCO(osp.join(self.data_path, 'EHF.json'))
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_shape = (img['height'], img['width'])
            img_path = osp.join(self.data_path, img['file_name'])

            bbox = ann['body_bbox']
            bbox = set_aspect_ratio(bbox, cfg.input_img_shape[1]/cfg.input_img_shape[0]) 
            vert_gt_path = osp.join(self.data_path, img['file_name'].split('_')[0] + '_align.ply')

            data_dict = {'img_path': img_path, 'img_shape': img_shape, 'bbox': bbox, 'vert_gt_path': vert_gt_path}
            datalist.append(data_dict)

        return datalist
 
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox, vert_gt_path = data['img_path'], data['img_shape'], data['bbox'], data['vert_gt_path']

        # image load
        img = load_img(img_path)

        # affine transform
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
        img = self.transform(img.astype(np.float32))/255.

        # vert gt load
        vert_gt, _ = load_ply(vert_gt_path)
    
        inputs = {'img': img}
        targets = {'smplx_vert_cam': vert_gt}
        meta_info = {'bb2img_trans': bb2img_trans}
        return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        eval_result = {'pa_mpvpe_all': [], 'pa_mpvpe_hand': [], 'pa_mpvpe_face': [], 'mpvpe_all': [], 'mpvpe_hand': [], 'mpvpe_face': []}
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            out = outs[n]
            
            # MPVPE from all vertices
            vert_gt = np.dot(self.cam_param['R'], out['smplx_vert_cam_target'].transpose(1,0)).transpose(1,0)
            vert_out = out['smplx_vert_cam']
            vert_out_align = rigid_align(vert_out, vert_gt)
            eval_result['pa_mpvpe_all'].append(np.sqrt(np.sum((vert_out_align - vert_gt)**2,1)).mean() * 1000)
            vert_out_align = vert_out - np.dot(smpl_x.vert_to_joint, vert_out)[smpl_x.vert_to_joint_idx['pelvis'],None,:] + np.dot(smpl_x.vert_to_joint, vert_gt)[smpl_x.vert_to_joint_idx['pelvis'],None,:]
            eval_result['mpvpe_all'].append(np.sqrt(np.sum((vert_out_align - vert_gt)**2,1)).mean() * 1000)


            # MPVPE from hand vertices
            vert_gt_lhand = vert_gt[smpl_x.hand_vertex_idx['left_hand'],:]
            vert_out_lhand = vert_out[smpl_x.hand_vertex_idx['left_hand'],:]
            vert_out_lhand_align = rigid_align(vert_out_lhand, vert_gt_lhand)
            vert_gt_rhand = vert_gt[smpl_x.hand_vertex_idx['right_hand'],:]
            vert_out_rhand = vert_out[smpl_x.hand_vertex_idx['right_hand'],:]
            vert_out_rhand_align = rigid_align(vert_out_rhand, vert_gt_rhand)
            eval_result['pa_mpvpe_hand'].append((np.sqrt(np.sum((vert_out_lhand_align - vert_gt_lhand)**2,1)).mean() * 1000 + np.sqrt(np.sum((vert_out_rhand_align - vert_gt_rhand)**2,1)).mean() * 1000)/2.)
            vert_out_lhand_align = vert_out_lhand - np.dot(smpl_x.vert_to_joint, vert_out)[smpl_x.vert_to_joint_idx['lwrist'],None,:] + np.dot(smpl_x.vert_to_joint, vert_gt)[smpl_x.vert_to_joint_idx['lwrist'],None,:]
            vert_out_rhand_align = vert_out_rhand - np.dot(smpl_x.vert_to_joint, vert_out)[smpl_x.vert_to_joint_idx['rwrist'],None,:] + np.dot(smpl_x.vert_to_joint, vert_gt)[smpl_x.vert_to_joint_idx['rwrist'],None,:]
            eval_result['mpvpe_hand'].append((np.sqrt(np.sum((vert_out_lhand_align - vert_gt_lhand)**2,1)).mean() * 1000 + np.sqrt(np.sum((vert_out_rhand_align - vert_gt_rhand)**2,1)).mean() * 1000)/2.)

            # MPVPE from face vertices
            vert_gt_face = vert_gt[smpl_x.face_vertex_idx,:]
            vert_out_face = vert_out[smpl_x.face_vertex_idx,:]
            vert_out_face_align = rigid_align(vert_out_face, vert_gt_face)
            eval_result['pa_mpvpe_face'].append(np.sqrt(np.sum((vert_out_face_align - vert_gt_face)**2,1)).mean() * 1000)
            vert_out_face_align = vert_out_face - np.dot(smpl_x.vert_to_joint, vert_out)[smpl_x.vert_to_joint_idx['neck'],None,:] + np.dot(smpl_x.vert_to_joint, vert_gt)[smpl_x.vert_to_joint_idx['neck'],None,:]
            eval_result['mpvpe_face'].append(np.sqrt(np.sum((vert_out_face_align - vert_gt_face)**2,1)).mean() * 1000)
            
        return eval_result

    def print_eval_result(self, eval_result):
        print('PA MPVPE (All): %.2f mm' % np.mean(eval_result['pa_mpvpe_all']))
        print('PA MPVPE (Hands): %.2f mm' % np.mean(eval_result['pa_mpvpe_hand']))
        print('PA MPVPE (Face): %.2f mm' % np.mean(eval_result['pa_mpvpe_face']))

        print('MPVPE (All): %.2f mm' % np.mean(eval_result['mpvpe_all']))
        print('MPVPE (Hands): %.2f mm' % np.mean(eval_result['mpvpe_hand']))
        print('MPVPE (Face): %.2f mm' % np.mean(eval_result['mpvpe_face']))


