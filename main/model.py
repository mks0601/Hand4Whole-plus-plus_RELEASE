import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.vit import ViT
from nets.resnet import ResNetBackbone
from nets.module import BodyPositionNet, BodyRotationNet, FaceRoI, FaceRegressor, HandRoI, HandControlNet
from nets.wilor import WiLoR_det, WiLoR
from nets.dwpose import DWPose
from nets.loss import PoseLoss, KptImgLoss, KptPelvisRelLoss, KptIHRelLoss, KptPartRelLoss, IHRootPoseReg, IHRelVecLoss
from utils.smpl_x import smpl_x
from utils.mano import mano
from pytorch3d.ops import corresponding_points_alignment
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from config import cfg
import copy

class Model(nn.Module):
    def __init__(self, encoder, body_position_net, body_rotation_net, face_roi_net, face_regressor, hand_control_net):
        super(Model, self).__init__()
        self.encoder = encoder
        self.body_position_net = body_position_net
        self.body_regressor = body_rotation_net
        self.face_roi_net = face_roi_net
        self.face_regressor = face_regressor
        self.hand_roi_net = HandRoI()
        self.wilor_det = WiLoR_det()
        self.wilor = WiLoR().to(dtype=torch.float16) # use half-precision to make it fast
        self.dwpose = DWPose()
        self.hand_control_net = hand_control_net
        self.smplx_layer = copy.deepcopy(smpl_x.layer['neutral']).cuda()

        self.pose_loss = PoseLoss()
        self.kpt_img_loss = KptImgLoss()
        self.kpt_pelvis_rel_loss = KptPelvisRelLoss()
        self.kpt_ih_rel_loss = KptIHRelLoss()
        self.kpt_part_rel_loss = KptPartRelLoss()
        self.ih_root_pose_reg = IHRootPoseReg()
        self.ih_rel_vec_loss = IHRelVecLoss()
        self.trainable_modules = [self.hand_control_net] 
        self.eval_modules = [self.wilor, self.dwpose, self.encoder, self.body_position_net, self.body_regressor, self.face_roi_net, self.face_regressor]
    
    def get_smplx_coord(self, root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, leye_pose, reye_pose, expr_param, shape_param, cam_trans):
        # camera-centered 3D coordinate
        output = self.smplx_layer(global_orient=root_pose, body_pose=body_pose, right_hand_pose=rhand_pose, left_hand_pose=lhand_pose, jaw_pose=jaw_pose, leye_pose=leye_pose, reye_pose=reye_pose, expression=expr_param, betas=shape_param, transl=cam_trans)
        vert_cam = output.vertices
        kpt_cam = output.joints[:,smpl_x.kpt['idx'],:]
        kpt_cam_orig = output.joints
        return vert_cam, kpt_cam, kpt_cam_orig
    
    def project_coord(self, xyz_cam):
        x = xyz_cam[:,:,0] / (xyz_cam[:,:,2] + 1e-4) * cfg.focal[0] + cfg.princpt[0]
        y = xyz_cam[:,:,1] / (xyz_cam[:,:,2] + 1e-4) * cfg.focal[1] + cfg.princpt[1]
        x = x / cfg.input_body_shape[1] * cfg.vit_output_shape[1]
        y = y / cfg.input_body_shape[0] * cfg.vit_output_shape[0]
        xy = torch.stack((x,y),2)
        return xy
    
    def combine_smplx_mano(self, smplx_vert_cam, smplx_kpt_cam, rmano_vert_cam, rmano_kpt_cam, rhand_root_pose, lmano_vert_cam, lmano_kpt_cam, lhand_root_pose):
        batch_size = smplx_vert_cam.shape[0]
        smplx_vert_cam_orig = smplx_vert_cam.clone()
        
        # hand outputs -> zero root pose space (right hand)
        rhand_root_pose = axis_angle_to_matrix(rhand_root_pose)
        rmano_vert_cam = torch.bmm(torch.inverse(rhand_root_pose), rmano_vert_cam.permute(0,2,1)).permute(0,2,1)
        rmano_kpt_cam = torch.bmm(torch.inverse(rhand_root_pose), rmano_kpt_cam.permute(0,2,1)).permute(0,2,1)
        # hand outputs -> zero root pose space (left hand)
        lhand_root_pose = axis_angle_to_matrix(lhand_root_pose)
        lmano_vert_cam = torch.bmm(torch.inverse(lhand_root_pose), lmano_vert_cam.permute(0,2,1)).permute(0,2,1)
        lmano_kpt_cam = torch.bmm(torch.inverse(lhand_root_pose), lmano_kpt_cam.permute(0,2,1)).permute(0,2,1)

        # zero root pose space -> smplx wrist space (right hand)
        RTs = corresponding_points_alignment(rmano_kpt_cam[:,mano.kpt['rigid_align_idx'],:], smplx_kpt_cam[:,smpl_x.kpt['hand_rigid_align_idx']['right'],:])
        R, t = RTs.R.permute(0,2,1), RTs.T
        rmano_vert_cam = torch.bmm(R, rmano_vert_cam.permute(0,2,1)).permute(0,2,1) + t.view(-1,1,3)
        rmano_kpt_cam = torch.bmm(R, rmano_kpt_cam.permute(0,2,1)).permute(0,2,1) + t.view(-1,1,3)
        rhand_root_pose = matrix_to_axis_angle(R)
        # zero root pose space -> smplx wrist space (left hand)
        RTs = corresponding_points_alignment(lmano_kpt_cam[:,mano.kpt['rigid_align_idx'],:], smplx_kpt_cam[:,smpl_x.kpt['hand_rigid_align_idx']['left'],:])
        R, t = RTs.R.permute(0,2,1), RTs.T
        lmano_vert_cam = torch.bmm(R, lmano_vert_cam.permute(0,2,1)).permute(0,2,1) + t.view(-1,1,3)
        lmano_kpt_cam = torch.bmm(R, lmano_kpt_cam.permute(0,2,1)).permute(0,2,1) + t.view(-1,1,3)
        lhand_root_pose = matrix_to_axis_angle(R)

        # combine smplx and mano coordinates (right hand)
        smplx_vert_cam = smplx_vert_cam.scatter_(1, torch.LongTensor(smpl_x.hand_vertex_idx['right_hand'])[None,:,None].repeat(batch_size,1,3).cuda(), rmano_vert_cam)
        smplx_idx = [smpl_x.kpt['name'].index('R_' + mano_name) for mano_name in mano.kpt['name']]
        smplx_kpt_cam = smplx_kpt_cam.scatter_(1, torch.LongTensor(smplx_idx)[None,:,None].repeat(batch_size,1,3).cuda(), rmano_kpt_cam)
        # combine smplx and mano coordinates (left hand)
        smplx_vert_cam = smplx_vert_cam.scatter_(1, torch.LongTensor(smpl_x.hand_vertex_idx['left_hand'])[None,:,None].repeat(batch_size,1,3).cuda(), lmano_vert_cam)
        smplx_idx = [smpl_x.kpt['name'].index('L_' + mano_name) for mano_name in mano.kpt['name']]
        smplx_kpt_cam = smplx_kpt_cam.scatter_(1, torch.LongTensor(smplx_idx)[None,:,None].repeat(batch_size,1,3).cuda(), lmano_kpt_cam)
        
        return smplx_vert_cam, smplx_kpt_cam, rhand_root_pose, lhand_root_pose
    
    def smooth_hand_boundary(self, smplx_vert, tot_itr=5, smooth_weight=0.5):
        smplx_vert = smplx_vert.clone()
        rhand_boundary_idx = torch.LongTensor(smpl_x.hand_boundary_idx['right']).cuda()
        lhand_boundary_idx = torch.LongTensor(smpl_x.hand_boundary_idx['left']).cuda()
        hand_boundary_idx = torch.cat((rhand_boundary_idx, lhand_boundary_idx))
        
        # include more neighbors
        vert_neighbor_idxs = smpl_x.vert_neighbor_idxs.long().cuda()
        for _ in range(3):
            new_boundary_idx = vert_neighbor_idxs[hand_boundary_idx].view(-1)
            new_boundary_idx = new_boundary_idx[new_boundary_idx != -1]
            hand_boundary_idx = torch.cat((hand_boundary_idx, new_boundary_idx))
        vert_neighbor_idxs = vert_neighbor_idxs[hand_boundary_idx]

        for _ in range(tot_itr):
            # gather all neighbor positions
            neighbor_positions = smplx_vert[:, vert_neighbor_idxs] # (batch_size, boundary_num, max_neighbor_num, 3)
            mask = (vert_neighbor_idxs != -1)[None, :, :, None] # (1, boundary_num, max_neighbor_num, 1)
            neighbor_positions = neighbor_positions * mask # zero out invalid entries

            # valid neighbor count per vertex
            num_valid = mask.sum(dim=2, keepdim=True).clamp(min=1)

            # average over neighbors
            neighbor_mean = neighbor_positions.sum(dim=2) / num_valid.squeeze(2)

            # only update boundary indices
            current = smplx_vert[:, hand_boundary_idx]
            updated = (1 - smooth_weight) * current + smooth_weight * neighbor_mean
            smplx_vert[:, hand_boundary_idx] = updated

        return smplx_vert

    def forward(self, inputs, targets, meta_info, mode):
        batch_size = inputs['img'].shape[0]
        
        # hand
        body_img = F.interpolate(inputs['img'], cfg.input_body_shape, mode='bilinear')
        with torch.no_grad():
            # for hand-only datasets
            #rhand_bbox, lhand_bbox, rhand_exist, lhand_exist = self.wilor_det(body_img)

            # for full-body datasets
            dwpose_kpt = self.dwpose(body_img) # batch_size, smpl_x.kpt['num'], 3 (x, y, score)
            rhand_bbox, lhand_bbox, rhand_exist, lhand_exist = self.dwpose.get_hand_bbox(dwpose_kpt)

            rhand_img, lhand_img = self.hand_roi_net(inputs['img'], rhand_bbox, lhand_bbox)
            rmano_vert_cam, rmano_kpt_cam, rhand_root_pose, rhand_pose, rhand_shape_param, rhand_transl, rhand_img_feat, lmano_vert_cam, lmano_kpt_cam, lhand_root_pose, lhand_pose, lhand_shape_param, lhand_transl, lhand_img_feat = self.wilor(rhand_img, lhand_img, rhand_bbox, lhand_bbox)
 
        # hand control net
        hand_feat_list = self.hand_control_net(rhand_img_feat, lhand_img_feat, rhand_bbox, lhand_bbox, rhand_exist, lhand_exist)

        # encoder
        img_feat, task_tokens = self.encoder(body_img, hand_feat_list)
        shape_token, cam_token, body_pose_token = task_tokens[:, 0], task_tokens[:, 1], task_tokens[:, 6:]

        # body
        body_kpt_img = self.body_position_net(img_feat)
        root_pose, body_pose, shape_param, cam_trans = self.body_regressor(body_pose_token, shape_token, cam_token, body_kpt_img.detach())

        # face
        face_bbox = self.face_roi_net.get_face_bbox(body_kpt_img.detach())
        face_feat = self.face_roi_net(inputs['img'], face_bbox)
        expr_param, jaw_pose = self.face_regressor(face_feat)

        # coordinates from smplx layer
        leye_pose, reye_pose = torch.zeros((batch_size,3)).float().cuda(), torch.zeros((batch_size,3)).float().cuda()
        vert_cam, kpt_cam, kpt_cam_orig = self.get_smplx_coord(root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, leye_pose, reye_pose, expr_param, shape_param, cam_trans)

        # combine smplx and mano
        # only for hand-only datasets
        vert_cam, kpt_cam, rhand_root_pose, lhand_root_pose = self.combine_smplx_mano(vert_cam, kpt_cam, rmano_vert_cam, rmano_kpt_cam, rhand_root_pose, lmano_vert_cam, lmano_kpt_cam, lhand_root_pose)
        if mode == 'test':
            vert_cam = self.smooth_hand_boundary(vert_cam)

        # project camera-centered 3D coordinates to the screen space
        kpt_proj = self.project_coord(kpt_cam)
        kpt_proj_orig = self.project_coord(kpt_cam_orig)

        # root-relative keypoint coordinates
        kpt_cam = kpt_cam - kpt_cam[:,smpl_x.kpt['root_idx'],None,:]
        
        if mode == 'train':
            # loss functions
            loss = {}
            pose = torch.cat((root_pose, body_pose, jaw_pose, leye_pose, reye_pose, lhand_pose, rhand_pose),1).view(batch_size,smpl_x.joint['num'],3) # follow smpl_x.joint['name']
            loss['smplx_pose'] = self.pose_loss(pose, targets['smplx_pose'], meta_info['smplx_pose_valid'])
            loss['smplx_shape'] = shape_param ** 2 * 0.01 # ih26m+reih+arctic
            #loss['smplx_shape'] = torch.abs(shape_param - targets['smplx_shape']) * meta_info['smplx_shape_valid'][:,None] + shape_param ** 2 * 0.001 * (1 - meta_info['smplx_shape_valid'][:,None]) # ih26m+reih+arctic+agora
            loss['mano_root_pose'] = self.pose_loss(rhand_root_pose, targets['rmano_root_pose'], meta_info['rmano_root_valid']) \
                                    + self.pose_loss(lhand_root_pose, targets['lmano_root_pose'], meta_info['lmano_root_valid'])
            loss['kpt_cam'] = (self.kpt_pelvis_rel_loss(kpt_cam, targets['kpt_cam'], meta_info['kpt_valid']*meta_info['is_3D'][:,None,None]) \
                    + self.kpt_ih_rel_loss(kpt_cam, targets['kpt_cam'], meta_info['kpt_valid']*meta_info['is_3D'][:,None,None]*meta_info['is_kpt_ih'][:,None,None]) \
                    + self.kpt_part_rel_loss(kpt_cam, targets['kpt_cam'], meta_info['kpt_valid']*meta_info['is_3D'][:,None,None])) * 10 
            loss['smplx_kpt_cam'] = (self.kpt_pelvis_rel_loss(kpt_cam, targets['smplx_kpt_cam'], meta_info['smplx_kpt_valid']) \
                    + self.kpt_ih_rel_loss(kpt_cam, targets['smplx_kpt_cam'], meta_info['smplx_kpt_valid']*meta_info['is_smplx_ih'][:,None,None]) \
                    + self.kpt_part_rel_loss(kpt_cam, targets['smplx_kpt_cam'], meta_info['smplx_kpt_valid'])) * 10 
            loss['kpt_proj'] = torch.abs(kpt_proj - targets['kpt_img']) * meta_info['kpt_valid']
            loss['kpt_img'] = self.kpt_img_loss(body_kpt_img, smpl_x.kpt_to_kpt_hm(targets['kpt_img']), smpl_x.kpt_to_kpt_hm(targets['kpt_cam']), smpl_x.kpt_to_kpt_hm(meta_info['kpt_trunc']), meta_info['is_3D'])
            loss['smplx_kpt_img'] = self.kpt_img_loss(body_kpt_img, smpl_x.kpt_to_kpt_hm(targets['smplx_kpt_img']), smpl_x.kpt_to_kpt_hm(targets['smplx_kpt_cam']), smpl_x.kpt_to_kpt_hm(meta_info['smplx_kpt_trunc']), torch.ones_like(meta_info['is_3D']))
            loss['ih_root_pose_reg'] = self.ih_root_pose_reg(pose, kpt_cam, meta_info['cam_R'], meta_info['cam_R_valid']) * meta_info['is_hand_only'] * (meta_info['smplx_pose_valid'][:,smpl_x.joint['root_idx'],0]==0)
            loss['ih_rel_vec'] = (self.ih_rel_vec_loss(kpt_cam, targets['kpt_cam'], meta_info['kpt_valid']*meta_info['is_kpt_ih'][:,None,None]) \
                                + self.ih_rel_vec_loss(kpt_cam, targets['smplx_kpt_cam'], meta_info['smplx_kpt_valid']*meta_info['is_smplx_ih'][:,None,None])) * 10
            return loss
        else:
            # test output
            out = {}
            out['img'] = inputs['img']
            out['kpt_img'] = body_kpt_img
            out['kpt_proj_orig'] = kpt_proj_orig
            out['rhand_bbox'] = rhand_bbox
            out['lhand_bbox'] = lhand_bbox
            out['rmano_vert_cam'] = rmano_vert_cam
            out['lmano_vert_cam'] = lmano_vert_cam
            out['smplx_kpt_proj'] = kpt_proj
            out['smplx_kpt_proj_orig'] = kpt_proj_orig
            out['smplx_kpt_cam'] = kpt_cam
            out['smplx_vert_cam'] = vert_cam
            out['smplx_root_pose'] = root_pose
            out['smplx_body_pose'] = body_pose
            out['smplx_lhand_pose'] = lhand_pose
            out['smplx_rhand_pose'] = rhand_pose
            out['smplx_jaw_pose'] = jaw_pose
            out['smplx_leye_pose'] = leye_pose
            out['smplx_reye_pose'] = reye_pose
            out['smplx_shape'] = shape_param
            out['smplx_expr'] = expr_param
            out['smplx_trans'] = cam_trans
            if 'smplx_vert_cam' in targets:
                out['smplx_vert_cam_target'] = targets['smplx_vert_cam']
            if 'rmano_vert_cam' in targets:
                out['rmano_vert_cam_target'] = targets['rmano_vert_cam']
            if 'lmano_vert_cam' in targets:
                out['lmano_vert_cam_target'] = targets['lmano_vert_cam']
            if 'kpt_cam' in targets:
                out['kpt_cam_target'] = targets['kpt_cam']
            if 'rmano_kpt_cam' in targets:
                out['rmano_kpt_cam_target'] = targets['rmano_kpt_cam']
            if 'lmano_kpt_cam' in targets:
                out['lmano_kpt_cam_target'] = targets['lmano_kpt_cam']
            if 'kpt_valid' in meta_info:
                out['kpt_valid'] = meta_info['kpt_valid']
            if 'bb2img_trans' in meta_info:
                out['bb2img_trans'] = meta_info['bb2img_trans']
            return out

def init_weights(m):
    try:
        if type(m) == nn.ConvTranspose2d:
            nn.init.normal_(m.weight,std=0.001)
        elif type(m) == nn.Conv2d:
            nn.init.normal_(m.weight,std=0.001)
            nn.init.constant_(m.bias, 0)
        elif type(m) == nn.BatchNorm2d:
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias,0)
        elif type(m) == nn.Linear:
            nn.init.normal_(m.weight,std=0.01)
            nn.init.constant_(m.bias,0)
    except AttributeError:
        pass

def get_model(mode):
    encoder = ViT(img_size=cfg.input_body_shape, patch_size=16, embed_dim=cfg.vit_feat_dim, depth=24, num_heads=16, ratio=1, use_checkpoint=False, mlp_ratio=4, qkv_bias=True, drop_path_rate=0.5)

    body_position_net = BodyPositionNet(cfg.vit_feat_dim)
    body_rotation_net = BodyRotationNet(cfg.vit_feat_dim)

    face_backbone = ResNetBackbone(18)
    face_roi_net = FaceRoI(face_backbone)
    face_regressor = FaceRegressor()

    hand_control_net = HandControlNet()

    if mode == 'train':
        body_position_net.apply(init_weights)
        body_rotation_net.apply(init_weights)
        
        face_roi_net.apply(init_weights)
        face_backbone.init_weights()
        face_regressor.apply(init_weights)
        
        hand_control_net.init_weights()

    model = Model(encoder, body_position_net, body_rotation_net, face_roi_net, face_regressor, hand_control_net)
    return model
