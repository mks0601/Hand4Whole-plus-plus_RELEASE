import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
import kornia
from nets.layer import make_conv_layers, make_linear_layers, CrossAttn
from utils.transforms import rotation_6d_to_axis_angle, soft_argmax_3d, restore_bbox
from utils.smpl_x import smpl_x
from utils.mano import mano
from config import cfg
import math

class BodyPositionNet(nn.Module):
    def __init__(self, feat_dim):
        super(BodyPositionNet, self).__init__()
        self.kpt_num = len(smpl_x.kpt_hm['part_idx']['body'])
        self.hm_shape = cfg.output_hm_shape
        self.conv = make_conv_layers([feat_dim, self.kpt_num*self.hm_shape[0]], kernel=1, stride=1, padding=0, bnrelu_final=False)

    def forward(self, img_feat):
        kpt_hm = self.conv(img_feat).view(-1, self.kpt_num, self.hm_shape[0], self.hm_shape[1], self.hm_shape[2])
        kpt_coord = soft_argmax_3d(kpt_hm)
        return kpt_coord

class BodyRotationNet(nn.Module):
    def __init__(self, feat_dim):
        super(BodyRotationNet, self).__init__()
        self.kpt_num = len(smpl_x.kpt_hm['part_idx']['body'])
        self.body_conv = make_linear_layers([feat_dim, 512], relu_final=False)
        self.root_pose_out = make_linear_layers([self.kpt_num*(512+3), 6], relu_final=False)
        self.body_pose_out = make_linear_layers([self.kpt_num*(512+3), (len(smpl_x.joint['part_idx']['body'])-1)*6], relu_final=False) # without root
        self.shape_out = make_linear_layers([feat_dim, smpl_x.shape_param_dim], relu_final=False)
        self.cam_out = make_linear_layers([feat_dim, 3], relu_final=False)

    def get_camera_trans(self, cam_param):
        t_xy = cam_param[:, :2]
        gamma = torch.sigmoid(cam_param[:, 2]) # apply sigmoid to make it positive
        k_value = torch.FloatTensor([math.sqrt(cfg.focal[0] * cfg.focal[1] * cfg.bbox_3d_size * cfg.bbox_3d_size / (cfg.input_body_shape[0] * cfg.input_body_shape[1]))]).cuda().view(-1)
        t_z = k_value * gamma
        cam_trans = torch.cat((t_xy, t_z[:, None]), 1)
        return cam_trans

    def forward(self, body_pose_token, shape_token, cam_token, kpt_coord):
        batch_size = body_pose_token.shape[0]

        # body pose
        body_pose_token = self.body_conv(body_pose_token)
        body_pose_token = torch.cat((body_pose_token, kpt_coord), 2)
        root_pose = self.root_pose_out(body_pose_token.view(batch_size, -1))
        body_pose = self.body_pose_out(body_pose_token.view(batch_size, -1)).view(batch_size,-1,6)
        root_pose = rotation_6d_to_axis_angle(root_pose)
        body_pose = rotation_6d_to_axis_angle(body_pose.view(-1,6)).reshape(batch_size,-1)

        # shape parameter
        shape_param = self.shape_out(shape_token)

        # camera translation
        cam_param = self.cam_out(cam_token)
        cam_trans = self.get_camera_trans(cam_param)
        return root_pose, body_pose, shape_param, cam_trans

class FaceRoI(nn.Module):
    def __init__(self, backbone):
        super(FaceRoI, self).__init__()
        self.backbone = backbone

    def get_bbox_center_size(self, kpt):
        x, y = kpt[:,0], kpt[:,1]
        xmin, ymin, xmax, ymax = torch.min(x), torch.min(y), torch.max(x), torch.max(y)
        center = torch.FloatTensor([(xmin+xmax)/2, (ymin+ymax)/2]).cuda()
        size = torch.FloatTensor([(xmax-xmin), (ymax-ymin)]).cuda() * 1.5
        for i in range(2):
            if size[i] == 0:
                size[i] = 1e-4
        return center, size

    def get_face_bbox(self, kpt):
        batch_size = kpt.shape[0]

        # get face keypoints and change from cfg.output_hm_shape to cfg.input_body_shape
        kpt = kpt[:,smpl_x.kpt_hm['part_idx']['face'],:2].clone()
        kpt[:,:,0] = kpt[:,:,0] / cfg.output_hm_shape[2] * cfg.input_body_shape[1]
        kpt[:,:,1] = kpt[:,:,1] / cfg.output_hm_shape[1] * cfg.input_body_shape[0]
        
        # get bbox
        face_bbox_center, face_bbox_size = [], []
        for i in range(batch_size):
            _face_bbox_center, _face_bbox_size = self.get_bbox_center_size(kpt[i])
            face_bbox_center.append(_face_bbox_center); face_bbox_size.append(_face_bbox_size);
        face_bbox_center, face_bbox_size = torch.stack(face_bbox_center), torch.stack(face_bbox_size)
        face_bbox = restore_bbox(face_bbox_center, face_bbox_size, cfg.input_face_shape[1]/cfg.input_face_shape[0], 1.5).detach()  # xyxy in cfg.input_body_shape space
        return face_bbox

    def forward(self, img, face_bbox):
        # batch_idx, xmin, ymin, xmax, ymax
        batch_size = face_bbox.shape[0]
        face_bbox = torch.cat((torch.arange(batch_size).float().cuda()[:,None], face_bbox),1)  
 
        # cfg input_body_shape -> cfg.input_img_shape
        face_bbox_roi = face_bbox.clone()
        face_bbox_roi[:,1] = face_bbox_roi[:,1] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
        face_bbox_roi[:,2] = face_bbox_roi[:,2] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
        face_bbox_roi[:,3] = face_bbox_roi[:,3] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
        face_bbox_roi[:,4] = face_bbox_roi[:,4] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
 
        face_img = torchvision.ops.roi_align(img, face_bbox_roi, cfg.input_hand_shape, aligned=False)
        face_feat = self.backbone(face_img)
        return face_feat

class FaceRegressor(nn.Module):
    def __init__(self):
        super(FaceRegressor, self).__init__()
        self.expr_out = make_linear_layers([512, smpl_x.expr_param_dim], relu_final=False)
        self.jaw_pose_out = make_linear_layers([512, 6], relu_final=False)

    def forward(self, img_feat):
        expr_param = self.expr_out(img_feat.mean((2,3))) # expression parameter
        jaw_pose = rotation_6d_to_axis_angle(self.jaw_pose_out(img_feat.mean((2,3))))
        return expr_param, jaw_pose

class HandRoI(nn.Module):
    def __init__(self):
        super(HandRoI, self).__init__()

    def forward(self, img, rhand_bbox, lhand_bbox):
        # batch_idx, xmin, ymin, xmax, ymax
        batch_size = rhand_bbox.shape[0]
        rhand_bbox = torch.cat((torch.arange(batch_size).float().cuda()[:,None], rhand_bbox),1)  
        lhand_bbox = torch.cat((torch.arange(batch_size).float().cuda()[:,None], lhand_bbox),1)
 
        # cfg input_body_shape -> cfg.input_img_shape
        rhand_bbox_roi = rhand_bbox.clone()
        rhand_bbox_roi[:,1] = rhand_bbox_roi[:,1] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
        rhand_bbox_roi[:,2] = rhand_bbox_roi[:,2] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
        rhand_bbox_roi[:,3] = rhand_bbox_roi[:,3] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
        rhand_bbox_roi[:,4] = rhand_bbox_roi[:,4] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
 
        # cfg input_body_shape -> cfg.input_img_shape
        lhand_bbox_roi = lhand_bbox.clone()
        lhand_bbox_roi[:,1] = lhand_bbox_roi[:,1] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
        lhand_bbox_roi[:,2] = lhand_bbox_roi[:,2] / cfg.input_body_shape[0] * cfg.input_img_shape[0]
        lhand_bbox_roi[:,3] = lhand_bbox_roi[:,3] / cfg.input_body_shape[1] * cfg.input_img_shape[1]
        lhand_bbox_roi[:,4] = lhand_bbox_roi[:,4] / cfg.input_body_shape[0] * cfg.input_img_shape[0]

        rhand_img = torchvision.ops.roi_align(img, rhand_bbox_roi, cfg.input_hand_shape, aligned=False)
        lhand_img = torchvision.ops.roi_align(img, lhand_bbox_roi, cfg.input_hand_shape, aligned=False)
        return rhand_img, lhand_img

class HandControlNet(nn.Module):
    def __init__(self, wilor_feat_dim=1280, vit_feat_dim=cfg.vit_feat_dim, vit_depth=24):
        super(HandControlNet, self).__init__()
        self.wilor_feat_dim = wilor_feat_dim
        self.vit_feat_dim = vit_feat_dim
        self.vit_depth = vit_depth

        # body ViT output shape
        self.down_factor = 16 # downsampling factor from the body ViT input image shape
        self.target_shape = (cfg.input_body_shape[0]//self.down_factor, cfg.input_body_shape[1]//self.down_factor) 
        
        # layers
        self.pos_embed = self.get_pos_embed(self.wilor_feat_dim, cfg.input_body_shape[0], cfg.input_body_shape[1])
        self.cross_attn_depth = 3
        self.cross_attn = nn.ModuleList([CrossAttn(self.wilor_feat_dim) for _ in range(self.cross_attn_depth)])
        self.zero_convs_rhand = nn.ModuleList([nn.Conv2d(self.wilor_feat_dim, self.vit_feat_dim, 1) for _ in range(self.vit_depth)])
        self.zero_convs_lhand = nn.ModuleList([nn.Conv2d(self.wilor_feat_dim, self.vit_feat_dim, 1) for _ in range(self.vit_depth)])


    def init_weights(self):
        for i in range(self.vit_depth):
            nn.init.constant_(self.zero_convs_rhand[i].weight, 0)
            nn.init.constant_(self.zero_convs_rhand[i].bias, 0)
            nn.init.constant_(self.zero_convs_lhand[i].weight, 0)
            nn.init.constant_(self.zero_convs_lhand[i].bias, 0)

    # https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/torch_encodings.py
    def get_pos_embed(self, channel_num, height, width):
        channel_num_orig = channel_num
        channel_num = math.ceil(channel_num / 4) * 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channel_num, 2).float() / channel_num)).float().cuda()

        pos_x = torch.arange(width).float().cuda()
        pos_y = torch.arange(height).float().cuda()
        sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq) # width, channel_num//2
        sin_inp_y = torch.einsum("i,j->ij", pos_y, inv_freq) # height, channel_num//2

        embed_x = torch.stack((sin_inp_x.cos(), sin_inp_x.sin()),2).permute(1,2,0).reshape(channel_num,1,width)
        embed_y = torch.stack((sin_inp_y.cos(), sin_inp_y.sin()),2).permute(1,2,0).reshape(channel_num,height,1)

        embed = torch.zeros((channel_num*2, height, width)).float().cuda()
        embed[:channel_num,:,:] = embed_x
        embed[channel_num:,:,:] = embed_y
        embed = embed[:channel_num_orig,:,:]
        return embed

    def crop_and_resize(self, feat, bbox, target_shape):
        batch_size = bbox.shape[0]
        transl = torch.FloatTensor([target_shape[1]/2, target_shape[0]/2]).cuda()[None,:] - (bbox[:,:2] + bbox[:,2:])/2
        center = (bbox[:,:2] + bbox[:,2:])/2
        scale = torch.stack([target_shape[1]/(bbox[:,2]-bbox[:,0]+1e-8), target_shape[0]/(bbox[:,3]-bbox[:,1]+1e-8)],1)
        angle = torch.zeros((batch_size)).float().cuda()
        affine_trans_mat = kornia.geometry.transform.get_affine_matrix2d(transl, center, scale, angle)[:,:2,:]

        feat = kornia.geometry.transform.warp_affine(feat, affine_trans_mat, target_shape)
        return feat

    def undo_crop_and_resize(self, feat, bbox, target_shape):
        batch_size, _, src_height, src_width = feat.shape

        center = torch.FloatTensor([src_width/2, src_height/2]).cuda()[None,:].repeat(batch_size,1)
        transl = (bbox[:,:2] + bbox[:,2:])/2 - center
        scale = torch.stack([(bbox[:,2]-bbox[:,0])/src_width, (bbox[:,3]-bbox[:,1])/src_height],1)
        angle = torch.zeros((batch_size)).float().cuda()
        affine_trans_mat = kornia.geometry.transform.get_affine_matrix2d(transl, center, scale, angle)[:,:2,:]

        feat = kornia.geometry.transform.warp_affine(feat, affine_trans_mat, target_shape)
        return feat

    def get_iou(self, box1, box2):
        box1 = box1.clone()
        box2 = box2.clone()
        box1 = box1.view(-1,4)
        box2 = box2.view(-1,4)
        box1[:,2:] += box1[:,:2] # xywh -> xyxy
        box2[:,2:] += box2[:,:2] # xywh -> xyxy

        xmin = torch.maximum(box1[:,0], box2[:,0])
        ymin = torch.maximum(box1[:,1], box2[:,1])
        xmax = torch.minimum(box1[:,2], box2[:,2])
        ymax = torch.minimum(box1[:,3], box2[:,3])
        inter_area = torch.maximum(torch.zeros_like(xmax), xmax - xmin) * torch.maximum(torch.zeros_like(ymax), ymax - ymin)

        box1_area = (box1[:,2] - box1[:,0]) * (box1[:,3] - box1[:,1])
        box2_area = (box2[:,2] - box2[:,0]) * (box2[:,3] - box2[:,1])
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / (union_area + 1e-5)
        return iou

    def forward(self, rhand_img_feat, lhand_img_feat, rhand_bbox, lhand_bbox, rhand_exist, lhand_exist):
        batch_size = rhand_img_feat.shape[0]
        
        # positional embedding
        rhand_pos_embed = self.crop_and_resize(self.pos_embed[None].repeat(batch_size,1,1,1), rhand_bbox, (cfg.input_hand_shape[0]//self.down_factor, cfg.input_hand_shape[1]//self.down_factor))
        lhand_pos_embed = self.crop_and_resize(self.pos_embed[None].repeat(batch_size,1,1,1), lhand_bbox, (cfg.input_hand_shape[0]//self.down_factor, cfg.input_hand_shape[1]//self.down_factor))
        rhand_pos_embed = rhand_pos_embed[:,:,:,2:-2]
        lhand_pos_embed = lhand_pos_embed[:,:,:,2:-2]
        
        # cross attention
        rhand_img_feat_orig, lhand_img_feat_orig, use_cross_attn = rhand_img_feat, lhand_img_feat, (rhand_exist*lhand_exist)[:,None,None,None]
        #iou = self.get_iou(rhand_bbox, lhand_bbox)
        #use_cross_attn = use_cross_attn*(iou[:,None,None,None]>0.5).float()
        rhand_img_feat = rhand_img_feat + rhand_pos_embed
        lhand_img_feat = lhand_img_feat + lhand_pos_embed
        rhand_img_feat = rhand_img_feat.view(batch_size,self.wilor_feat_dim,self.target_shape[0]*self.target_shape[1]).permute(0,2,1)
        lhand_img_feat = lhand_img_feat.view(batch_size,self.wilor_feat_dim,self.target_shape[0]*self.target_shape[1]).permute(0,2,1)
        for i in range(self.cross_attn_depth):
            rhand_img_feat, lhand_img_feat = self.cross_attn[i](rhand_img_feat, lhand_img_feat)
        rhand_img_feat = rhand_img_feat.permute(0,2,1).reshape(batch_size,self.wilor_feat_dim,self.target_shape[0],self.target_shape[1])
        lhand_img_feat = lhand_img_feat.permute(0,2,1).reshape(batch_size,self.wilor_feat_dim,self.target_shape[0],self.target_shape[1])
        rhand_img_feat = rhand_img_feat*use_cross_attn + rhand_img_feat_orig*(1-use_cross_attn)
        lhand_img_feat = lhand_img_feat*use_cross_attn + lhand_img_feat_orig*(1-use_cross_attn)

        # process with zero-initialized convolutions
        rhand_feat_list = [self.zero_convs_rhand[i](rhand_img_feat)*rhand_exist[:,None,None,None] for i in range(self.vit_depth)]
        lhand_feat_list = [self.zero_convs_lhand[i](lhand_img_feat)*lhand_exist[:,None,None,None] for i in range(self.vit_depth)]

        # WiLoR takes (256,192) after cropping the (256,256) input image, so img_feat has the shape of (16,12)
        # restore (16,12) to (16,16) with the zero padding
        rhand_feat_list = [F.pad(x, (2,2)) for x in rhand_feat_list]
        lhand_feat_list = [F.pad(y, (2,2)) for y in lhand_feat_list]

        # change bbox from cfg.input_body_shape to the body ViT output shape
        rhand_bbox, lhand_bbox = rhand_bbox / self.down_factor, lhand_bbox / self.down_factor

        # undo crop and resize (restore from cfg.input_hand_shape to cfg.input_body_shape)
        #rhand_feat_list = [self.undo_crop_and_resize(x, rhand_bbox, self.target_shape) for x in rhand_feat_list]
        #lhand_feat_list = [self.undo_crop_and_resize(y, lhand_bbox, self.target_shape) for y in lhand_feat_list]
        rhand_feat_list = torch.chunk(self.undo_crop_and_resize(torch.cat(rhand_feat_list,1), rhand_bbox, self.target_shape), self.vit_depth, dim=1)
        lhand_feat_list = torch.chunk(self.undo_crop_and_resize(torch.cat(lhand_feat_list,1), lhand_bbox, self.target_shape), self.vit_depth, dim=1)
       
        # use maximal values
        hand_feat_list = [torch.maximum(x,y) for x,y in zip(rhand_feat_list, lhand_feat_list)]
        return hand_feat_list


