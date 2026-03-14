import torch
import torch.nn as nn
import os
import os.path as osp
import numpy as np
from wilor.models import WiLoR, load_wilor
from wilor.utils.renderer import cam_crop_to_full
from ultralytics import YOLO 
from pytorch3d.transforms import matrix_to_axis_angle
from utils.transforms import restore_bbox
from utils.mano import mano
from config import cfg

class WiLoR_det(nn.Module):
    def __init__(self):
        super(WiLoR_det, self).__init__()
        os.chdir(cfg.wilor_root_path)
        self.detector = YOLO('./pretrained_models/detector.pt')
        os.chdir(cfg.cur_dir)
    
    def forward(self, img):
        batch_size = img.shape[0]
        
        # forward to the hand detector of WiLoR
        # detect two hands with the highest confidence
        rhand_bbox, lhand_bbox = [[0,0,1,1] for _ in range(batch_size)], [[0,0,1,1] for _ in range(batch_size)] # initialize with dummy boxes
        rhand_exist, lhand_exist = [], []
        for i in range(batch_size):
            img_cv2 = img[i].detach().cpu().numpy().transpose(1,2,0)[:,:,::-1]*255
            detections = self.detector(img_cv2, conf=0.3, verbose=False)[0]
            rhand_bbox_i, rhand_score_i = None, -1
            lhand_bbox_i, lhand_score_i = None, -1
            for det in detections: 
                is_rhand = det.boxes.cls.cpu().detach().squeeze().item()
                bbox_i = det.boxes.data.cpu().detach().squeeze().numpy()[:4] # xyxy
                score_i = det.boxes.conf.cpu().detach().squeeze().numpy()
                if is_rhand and (rhand_score_i < score_i):
                    rhand_bbox_i = bbox_i
                    rhand_score_i = score_i
                elif (not is_rhand) and (lhand_score_i < score_i):
                    lhand_bbox_i = bbox_i
                    lhand_score_i = score_i
            if rhand_bbox_i is not None:
                rhand_bbox[i] = rhand_bbox_i.tolist()
                rhand_exist.append(1)
            else:
                rhand_exist.append(0)
            if lhand_bbox_i is not None:
                lhand_bbox[i] = lhand_bbox_i.tolist()
                lhand_exist.append(1)
            else:
                lhand_exist.append(0)
        rhand_bbox = torch.FloatTensor(rhand_bbox).cuda().view(batch_size,4)
        lhand_bbox = torch.FloatTensor(lhand_bbox).cuda().view(batch_size,4)
        rhand_exist = torch.FloatTensor(rhand_exist).cuda()
        lhand_exist = torch.FloatTensor(lhand_exist).cuda()
       
        # decompose xyxy to center and size
        rhand_bbox_center, rhand_bbox_size = (rhand_bbox[:,:2] + rhand_bbox[:,2:])/2., (rhand_bbox[:,2:] - rhand_bbox[:,:2])
        lhand_bbox_center, lhand_bbox_size = (lhand_bbox[:,:2] + lhand_bbox[:,2:])/2., (lhand_bbox[:,2:] - lhand_bbox[:,:2])
        
        # extend boxes while preserving the aspect ratio
        rhand_bbox = restore_bbox(rhand_bbox_center, rhand_bbox_size, cfg.input_hand_shape[1]/cfg.input_hand_shape[0], 2.0).detach()  # xyxy in cfg.input_body_shape space
        lhand_bbox = restore_bbox(lhand_bbox_center, lhand_bbox_size, cfg.input_hand_shape[1]/cfg.input_hand_shape[0], 2.0).detach()  # xyxy in cfg.input_body_shape space
        return rhand_bbox, lhand_bbox, rhand_exist, lhand_exist

class WiLoR(nn.Module):
    def __init__(self):
        super(WiLoR, self).__init__()
        os.chdir(cfg.wilor_root_path)
        self.model, _ = load_wilor(checkpoint_path='./pretrained_models/wilor_final.ckpt', cfg_path='./pretrained_models/model_config.yaml')
        os.chdir(cfg.cur_dir)
        self.rgb_mean = (0.485, 0.456, 0.406)
        self.rgb_std = (0.229, 0.224, 0.225)
    
    def forward(self, rhand_img, lhand_img, rhand_bbox, lhand_bbox):
        batch_size = rhand_img.shape[0]
        
        # combine right and left hand images after flipping the left hand images
        img = torch.cat((rhand_img, torch.flip(lhand_img,[3]))) # right hand + flipped left hnad images
        bbox = torch.cat((rhand_bbox, lhand_bbox))
        is_lhand = torch.cat((torch.zeros((batch_size)), torch.ones((batch_size)))).float().cuda() == 1
        
        # forward to WiLoR
        img = img - torch.FloatTensor(self.rgb_mean).cuda().view(1,3,1,1)
        img = img / torch.FloatTensor(self.rgb_std).cuda().view(1,3,1,1)
        img = img.to(dtype=torch.float16) # use half-precision input for fast inference
        out = self.model({'img': img})
        
        # get camera translation (restore flipped left hands)
        pred_cam = out['pred_cam']
        pred_cam[is_lhand,1] = -pred_cam[is_lhand,1]
        box_center = (bbox[:,:2] + bbox[:,2:])/2.
        box_size = bbox[:,2] - bbox[:,0]
        img_size = torch.FloatTensor([cfg.input_body_shape[1],cfg.input_body_shape[0]]).view(1,2).cuda().repeat(batch_size*2,1)
        scaled_focal_length = 5000 / cfg.input_hand_shape[0] * max(cfg.input_body_shape)
        transl = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length)
        rhand_transl, lhand_transl = transl[is_lhand==0].float(), transl[is_lhand==1].float()
        
        # get MANO vertices (restore flipped left hands)
        vert_cam = out['pred_vertices'] 
        vert_cam[is_lhand,:,0] = -vert_cam[is_lhand,:,0]
        vert_cam = vert_cam + transl.view(-1,1,3)
        rhand_vert_cam, lhand_vert_cam = vert_cam[is_lhand==0].float(), vert_cam[is_lhand==1].float()
        rhand_kpt_cam = torch.bmm(torch.from_numpy(mano.kpt['regressor']).cuda()[None,:,:].repeat(batch_size,1,1), rhand_vert_cam).float()
        lhand_kpt_cam = torch.bmm(torch.from_numpy(mano.kpt['regressor']).cuda()[None,:,:].repeat(batch_size,1,1), lhand_vert_cam).float()
       
        # get MANO parameters (restore flipped left hands)
        mano_param = out['pred_mano_params']
        root_pose = matrix_to_axis_angle(mano_param['global_orient'])
        hand_pose = matrix_to_axis_angle(mano_param['hand_pose'])
        shape_param = mano_param['betas']
        root_pose[is_lhand,:,1:3] = -root_pose[is_lhand,:,1:3]
        hand_pose[is_lhand,:,1:3] = -hand_pose[is_lhand,:,1:3]
        rhand_root_pose, lhand_root_pose = root_pose[is_lhand==0].view(batch_size,3).float(), root_pose[is_lhand==1].view(batch_size,3).float()
        rhand_pose, lhand_pose = hand_pose[is_lhand==0].view(batch_size,-1).float(), hand_pose[is_lhand==1].view(batch_size,-1).float()
        rhand_shape_param, lhand_shape_param = shape_param[is_lhand==0].float(), shape_param[is_lhand==1].float()
        rhand_pose -= mano.layer['right'].pose_mean.float().cuda().view(1,-1)[:,3:]
        lhand_pose -= mano.layer['left'].pose_mean.float().cuda().view(1,-1)[:,3:]

        # get image feature (restore flipped left hands)
        img_feat = out['img_feat'].float()
        rhand_img_feat = img_feat[is_lhand==0]
        lhand_img_feat = torch.flip(img_feat[is_lhand==1], [3])

        return rhand_vert_cam, rhand_kpt_cam, rhand_root_pose, rhand_pose, rhand_shape_param, rhand_transl, rhand_img_feat, lhand_vert_cam, lhand_kpt_cam, lhand_root_pose, lhand_pose, lhand_shape_param, lhand_transl, lhand_img_feat
