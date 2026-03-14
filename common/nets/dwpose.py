import torch
import torch.nn as nn
import numpy as np
from mmpose.apis import init_model
from mmpose.structures import PoseDataSample
from mmengine.structures import InstanceData
from utils.smpl_x import smpl_x
from utils.transforms import restore_bbox
from config import cfg

class DWPose(nn.Module):
    def __init__(self):
        super(DWPose, self).__init__()
        self.kpt = {
                    'num': 133, # body 23, face 68, lhand 21, rhand 21
                    'name': ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', # body
                        *['Face_' + str(i) for i in range(52,69)], # face contour
                        *['Face_' + str(i) for i in range(1,52)], # face
                        'L_Wrist_Hand', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4', # left hand
                        'R_Wrist_Hand', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4') # right hand
                    }
        self.kpt['part_idx'] = {'body': range(self.kpt['name'].index('Nose'), self.kpt['name'].index('R_Heel')+1),
                            'lhand': range(self.kpt['name'].index('L_Wrist_Hand'), self.kpt['name'].index('L_Pinky_4')+1),
                            'rhand': range(self.kpt['name'].index('R_Wrist_Hand'), self.kpt['name'].index('R_Pinky_4')+1)}
        self.model = init_model(cfg.dwpose_path['cfg'], cfg.dwpose_path['ckpt'], device='cuda')
        self.model.cfg.model['test_cfg']['flip_test'] = False

    def get_bbox_center_size(self, kpt):
        x, y = kpt[:,0], kpt[:,1]
        xmin, ymin, xmax, ymax = torch.min(x), torch.min(y), torch.max(x), torch.max(y)
        center = torch.FloatTensor([(xmin+xmax)/2, (ymin+ymax)/2]).cuda()
        size = torch.FloatTensor([(xmax-xmin), (ymax-ymin)]).cuda() * 1.2
        for i in range(2):
            if size[i] == 0:
                size[i] = 1e-4
        return center, size

    def get_hand_bbox(self, kpt):
        batch_size = kpt.shape[0]
        kpt, kpt_valid = kpt[:,:,:2].clone(), (kpt[:,:,2:] > 0.3).float()
       
        rhand_idx = list(smpl_x.kpt['part_idx']['rhand']) + [smpl_x.kpt['name'].index('R_Wrist')]
        rhand_kpt, rhand_kpt_valid = kpt[:,rhand_idx,:], kpt_valid[:,rhand_idx,:]
        lhand_idx = list(smpl_x.kpt['part_idx']['lhand']) + [smpl_x.kpt['name'].index('L_Wrist')]
        lhand_kpt, lhand_kpt_valid = kpt[:,lhand_idx,:], kpt_valid[:,lhand_idx,:]

        # right hand
        rhand_bbox_center, rhand_bbox_size, rhand_exist = [], [], []
        for i in range(batch_size):
            if rhand_kpt_valid[i].sum() > 3:
                rhand_exist.append(1)
                _rhand_bbox_center, _rhand_bbox_size = self.get_bbox_center_size(rhand_kpt[i,rhand_kpt_valid[i,:,0]==1,:])
            else:
                rhand_exist.append(0)
                _rhand_bbox_center, _rhand_bbox_size = torch.ones((2)).float().cuda()*0.5, torch.ones((2)).float().cuda()
            rhand_bbox_center.append(_rhand_bbox_center); rhand_bbox_size.append(_rhand_bbox_size);
        rhand_bbox_center, rhand_bbox_size = torch.stack(rhand_bbox_center), torch.stack(rhand_bbox_size)
        rhand_bbox = restore_bbox(rhand_bbox_center, rhand_bbox_size, cfg.input_hand_shape[1]/cfg.input_hand_shape[0], 2.0).detach()  # xyxy in cfg.input_body_shape space
        rhand_exist = torch.FloatTensor(rhand_exist).cuda()

        # left hand
        lhand_bbox_center, lhand_bbox_size, lhand_exist = [], [], []
        for i in range(batch_size):
            if lhand_kpt_valid[i].sum() > 3:
                lhand_exist.append(1)
                _lhand_bbox_center, _lhand_bbox_size = self.get_bbox_center_size(lhand_kpt[i,lhand_kpt_valid[i,:,0]==1,:])
            else:
                lhand_exist.append(0)
                _lhand_bbox_center, _lhand_bbox_size = torch.ones((2)).float().cuda()*0.5, torch.ones((2)).float().cuda()
            lhand_bbox_center.append(_lhand_bbox_center); lhand_bbox_size.append(_lhand_bbox_size);
        lhand_bbox_center, lhand_bbox_size = torch.stack(lhand_bbox_center), torch.stack(lhand_bbox_size)
        lhand_bbox = restore_bbox(lhand_bbox_center, lhand_bbox_size, cfg.input_hand_shape[1]/cfg.input_hand_shape[0], 2.0).detach()  # xyxy in cfg.input_body_shape space
        lhand_exist = torch.FloatTensor(lhand_exist).cuda()

        return rhand_bbox, lhand_bbox, rhand_exist, lhand_exist

    def forward(self, img):
        batch_size = img.shape[0]
         
        # inference with pre-trained DWPose
        data_sample = PoseDataSample(gt_instances=InstanceData(bboxes=np.array([0,0,cfg.input_body_shape[1],cfg.input_body_shape[0]], dtype=np.float32)[None], bbox_scores=np.ones((1), dtype=np.float32)), 
                                    metainfo={'input_size': np.array(cfg.input_body_shape, dtype=np.float32), 'input_scale': np.array(cfg.input_body_shape, dtype=np.float32), 'input_center': np.array(cfg.input_body_shape, dtype=np.float32)/2})
        img = torch.flip(img, [1])*255 # RGB -> BGR, [0,1] -> [0,255]
        with torch.no_grad():
            batch_results = self.model.test_step({'inputs': img, 'data_samples': [data_sample]*batch_size})
        kpt = torch.from_numpy(np.stack([batch_results[i].pred_instances.keypoints[0] for i in range(batch_size)])).float().cuda() # batch_size, self.kpt['num'], 2
        score = torch.from_numpy(np.stack([batch_results[i].pred_instances.keypoint_scores[0] for i in range(batch_size)])).float().cuda() # batch_size, self.kpt['num']
        kpt = torch.cat((kpt, score[:,:,None]),2)

        # change keypoints from self.kpt['name'] to smpl_x.kpt['name']
        kpt_smplx = torch.zeros((batch_size,smpl_x.kpt['num'],3)).float().cuda()
        for dwpose_idx in range(self.kpt['num']):
            name = self.kpt['name'][dwpose_idx]
            if name in smpl_x.kpt['name']:
                smplx_idx = smpl_x.kpt['name'].index(name)
                kpt_smplx[:,smplx_idx,:] = kpt[:,dwpose_idx,:]
        
        """
        # for debug
        import cv2
        import random
        for i in range(batch_size):
            img_vis = img[i].detach().cpu().numpy().transpose(1,2,0).astype(np.uint8)
            kpt_vis = kpt[i].detach().cpu().numpy()
            score_vis = score[i].detach().cpu().numpy()
            for j in range(len(kpt_vis)):
                #if (self.kpt['name'][j] in ['L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip']) or (j in self.kpt['part_idx']['lhand']) or (j in self.kpt['part_idx']['rhand']):
                #    color = 255*kpt_vis[j][2]
                #    img_vis = cv2.circle(img_vis.copy(), (int(kpt_vis[j][0]), int(kpt_vis[j][1])), 3, (color,0,0), thickness=-1)
                if score_vis[j] > 0.3:
                    img_vis = cv2.circle(img_vis.copy(), (int(kpt_vis[j][0]), int(kpt_vis[j][1])), 3, (255,0,0), thickness=-1)
            cv2.imwrite(str(random.randint(1,500)) + '.jpg', img_vis)
            print('saved')
        """

        return kpt_smplx

