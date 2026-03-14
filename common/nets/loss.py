import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.smpl_x import smpl_x
from utils.mano import mano
from pytorch3d.transforms import axis_angle_to_matrix
from config import cfg

class PoseLoss(nn.Module):
    def __init__(self):
        super(PoseLoss, self).__init__()

    def forward(self, pose_out, pose_gt, pose_valid, norm='l1'):
        batch_size = pose_out.shape[0]

        pose_out = pose_out.view(batch_size,-1,3)
        pose_gt = pose_gt.view(batch_size,-1,3)

        pose_out = axis_angle_to_matrix(pose_out)
        pose_gt = axis_angle_to_matrix(pose_gt)
        
        joint_num = pose_out.shape[1]
        pose_valid = pose_valid.view(batch_size,joint_num,1,1)
        if norm == 'l1':
            loss = torch.abs(pose_out - pose_gt) * pose_valid
        elif norm == 'l2':
            loss = (pose_out - pose_gt) ** 2 * pose_valid
        return loss

class KptImgLoss(nn.Module):
    def __init__(self):
        super(KptImgLoss, self).__init__()
    
    def forward(self, kpt_img_out, kpt_img_gt, kpt_cam_gt, kpt_trunc, is_3D):
        root_idx = smpl_x.kpt_hm['root_idx']
        body_idx = smpl_x.kpt_hm['part_idx']['body']
       
        # make gt 2.5D coordinates in cfg.output_hm_shape space
        kpt_img_gt = torch.cat((kpt_img_gt, kpt_cam_gt[:,:,2,None]),2) 
        kpt_img_gt[:,:,0] = kpt_img_gt[:,:,0] / cfg.vit_output_shape[1] * cfg.output_hm_shape[2]
        kpt_img_gt[:,:,1] = kpt_img_gt[:,:,1] / cfg.vit_output_shape[0] * cfg.output_hm_shape[1]
        kpt_img_gt[:,:,2] = kpt_img_gt[:,:,2] - kpt_img_gt[:,root_idx,None,2] # root-relative depth
        kpt_img_gt[:,:,2] = (kpt_img_gt[:,:,2] / (cfg.body_3d_size/2) + 1) / 2. * cfg.output_hm_shape[0] # depth discretize
        kpt_img_gt = kpt_img_gt[:,body_idx,:]
        
        # make truncation mask
        kpt_trunc = kpt_trunc.repeat(1,1,3).clone() # x,y,z
        kpt_trunc[:,:,2] = kpt_trunc[:,:,2] * kpt_trunc[:,root_idx,None,2]
        kpt_trunc = kpt_trunc[:,body_idx,:]

        # calculate loss
        loss = torch.abs(kpt_img_out - kpt_img_gt) * kpt_trunc
        loss_z = loss[:,:,2:] * is_3D[:,None,None].float()
        loss = torch.cat((loss[:,:,:2], loss_z),2)
        return loss

class KptPelvisRelLoss(nn.Module):
    def __init__(self):
        super(KptPelvisRelLoss, self).__init__()
   
    def forward(self, kpt_out, kpt_gt, kpt_valid):
        loss = torch.abs(kpt_out - kpt_gt) * kpt_valid * kpt_valid[:,smpl_x.kpt['root_idx'],None,:]
        return loss

class KptIHRelLoss(nn.Module):
    def __init__(self):
        super(KptIHRelLoss, self).__init__()
    
    def make_relative_kpt(self, kpt, kpt_valid=None):
        # right hand root (right wrist)-relative coordinates
        rwrist_idx = smpl_x.kpt['name'].index('R_Wrist')
        lwrist_idx = smpl_x.kpt['name'].index('L_Wrist')
        kpt = kpt - kpt[:,rwrist_idx,None,:] 
        if kpt_valid is not None:
            kpt_valid = kpt_valid * kpt_valid[:,rwrist_idx,None,:] * kpt_valid[:,lwrist_idx,None,:]
        return kpt, kpt_valid
 
    def forward(self, kpt_out, kpt_gt, kpt_valid):
        kpt_out, _ = self.make_relative_kpt(kpt_out)
        kpt_gt, kpt_valid = self.make_relative_kpt(kpt_gt, kpt_valid)
        loss = torch.abs(kpt_out - kpt_gt) * kpt_valid 
        return loss

class KptPartRelLoss(nn.Module):
    def __init__(self):
        super(KptPartRelLoss, self).__init__()
   
    def make_relative_kpt(self, kpt, kpt_valid=None):
        # left hand root (left wrist)-relative coordinates
        lhand_idx = smpl_x.kpt['part_idx']['lhand']
        lwrist_idx = smpl_x.kpt['name'].index('L_Wrist')
        lhand = kpt[:,lhand_idx,:]
        lwrist = kpt[:,lwrist_idx,None,:]
        lhand = lhand - lwrist
        kpt = torch.cat((kpt[:,:lhand_idx[0],:], lhand, kpt[:,lhand_idx[-1]+1:,:]),1)
        if kpt_valid is not None:
            kpt_valid = torch.cat((kpt_valid[:,:lhand_idx[0],:], kpt_valid[:,lhand_idx,:]*kpt_valid[:,lwrist_idx,None,:], kpt_valid[:,lhand_idx[-1]+1:,:]),1)

        # right hand root (right wrist)-relative coordinates
        rhand_idx = smpl_x.kpt['part_idx']['rhand']
        rwrist_idx = smpl_x.kpt['name'].index('R_Wrist')
        rhand = kpt[:,rhand_idx,:]
        rwrist = kpt[:,rwrist_idx,None,:]
        rhand = rhand - rwrist
        kpt = torch.cat((kpt[:,:rhand_idx[0],:], rhand, kpt[:,rhand_idx[-1]+1:,:]),1)
        if kpt_valid is not None:
            kpt_valid = torch.cat((kpt_valid[:,:rhand_idx[0],:], kpt_valid[:,rhand_idx,:]*kpt_valid[:,rwrist_idx,None,:], kpt_valid[:,rhand_idx[-1]+1:,:]),1)

        # face root (neck)-relative coordinates
        face_idx = smpl_x.kpt['part_idx']['face']
        neck_idx = smpl_x.kpt['name'].index('Neck')
        face = kpt[:,face_idx,:]
        neck = kpt[:,neck_idx,None,:]
        face = face - neck
        kpt = torch.cat((kpt[:,:face_idx[0],:], face, kpt[:,face_idx[-1]+1:,:]),1)
        if kpt_valid is not None:
            kpt_valid = torch.cat((kpt_valid[:,:face_idx[0],:], kpt_valid[:,face_idx,:]*kpt_valid[:,neck_idx,None,:], kpt_valid[:,face_idx[-1]+1:,:]),1)

        # body root (pelvis)-relative coordinates 
        # body keypoints are already root-relaive
        # this should be done after the above hand and face parts as it affects lwrist_valid, rwrist_valid, and neck_valid (body_idx includes wrists and neck) -> kpt_valid of all hands and face could be zero
        body_idx = smpl_x.kpt['part_idx']['body']
        pelvis_idx = smpl_x.kpt['name'].index('Pelvis')
        if kpt_valid is not None:
            kpt_valid = torch.cat((kpt_valid[:,:body_idx[0],:], kpt_valid[:,body_idx,:]*kpt_valid[:,pelvis_idx,None,:], kpt_valid[:,body_idx[-1]+1:,:]),1)

        return kpt, kpt_valid

    def forward(self, kpt_out, kpt_gt, kpt_valid):
        kpt_out, _ = self.make_relative_kpt(kpt_out)
        kpt_gt, kpt_valid = self.make_relative_kpt(kpt_gt, kpt_valid)
        loss = torch.abs(kpt_out - kpt_gt) * kpt_valid 
        return loss

class IHRootPoseReg(nn.Module):
    def __init__(self):
        super(IHRootPoseReg, self).__init__()
    
    def forward(self, pose, kpt, cam_R, cam_R_valid):
        kpt = torch.bmm(torch.inverse(cam_R), kpt.permute(0,2,1)).permute(0,2,1) # camera coordimate -> world coordinate. ignore translation.
        pelvis_neck_vec = F.normalize(kpt[:,smpl_x.kpt['name'].index('Neck'),:] - kpt[:,smpl_x.kpt['name'].index('Pelvis'),:], p=2, dim=1)
        up_vec = torch.FloatTensor([0,-1,0]).cuda().view(1,3) # up direction in the world coordinate system
        loss = torch.abs((pelvis_neck_vec*up_vec).sum(1) - 1) * cam_R_valid
        return loss

class IHRelVecLoss(nn.Module):
    def __init__(self):
        super(IHRelVecLoss, self).__init__()
   
    def forward(self, kpt_out, kpt_gt, kpt_valid):
        rhand_idx = [smpl_x.kpt['name'].index('R_Wrist')] + list(smpl_x.kpt['part_idx']['rhand'])
        lhand_idx = [smpl_x.kpt['name'].index('L_Wrist')] + list(smpl_x.kpt['part_idx']['lhand'])
        vec_out = kpt_out[:,rhand_idx,None,:] - kpt_out[:,None,lhand_idx,:]
        vec_gt = kpt_gt[:,rhand_idx,None,:] - kpt_gt[:,None,lhand_idx,:]
        vec_valid = kpt_valid[:,rhand_idx,None,:] * kpt_valid[:,None,lhand_idx,:]
        loss = torch.abs(vec_out - vec_gt) * vec_valid
        return loss
