import numpy as np
import cv2
import random
from config import cfg
import math
from utils.smpl_x import smpl_x
from utils.smpl import smpl
from utils.mano import mano
from utils.transforms import cam2img, distort_projection_fisheye, change_kpt_name
from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_matrix
import torch

def load_img(path, order='RGB'):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order=='RGB':
        img = img[:,:,::-1].copy()
    
    img = img.astype(np.float32)
    return img

def get_bbox(kpt_img, kpt_valid, extend_ratio=1.2):
    x_img, y_img = kpt_img[:,0], kpt_img[:,1]
    x_img = x_img[kpt_valid==1]; y_img = y_img[kpt_valid==1];
    xmin = min(x_img); ymin = min(y_img); xmax = max(x_img); ymax = max(y_img);

    x_center = (xmin+xmax)/2.; width = xmax-xmin;
    xmin = x_center - 0.5 * width * extend_ratio
    xmax = x_center + 0.5 * width * extend_ratio
    
    y_center = (ymin+ymax)/2.; height = ymax-ymin;
    ymin = y_center - 0.5 * height * extend_ratio
    ymax = y_center + 0.5 * height * extend_ratio

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox

def set_aspect_ratio(bbox, aspect_ratio, extend_ratio=1.25):
    bbox = np.array(bbox, dtype=np.float32)

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w*extend_ratio
    bbox[3] = h*extend_ratio
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.
    
    bbox = bbox.astype(np.float32)
    return bbox

def get_aug_config():
    scale_factor = 0.25
    rot_factor = 30
    color_factor = 0.2
    
    scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
    rot = np.clip(np.random.randn(), -2.0, 2.0) * rot_factor if random.random() <= 0.6 else 0
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])
    do_flip = random.random() <= 0.5

    return scale, rot, color_scale, do_flip

def augmentation(img, bbox, data_split):
    if data_split == 'train':
        scale, rot, color_scale, do_flip = get_aug_config()
    else:
        scale, rot, color_scale, do_flip = 1.0, 0.0, np.array([1,1,1]), False
    
    img, trans, inv_trans = get_patch_img(img, bbox, scale, rot, do_flip, cfg.input_img_shape)
    img = np.clip(img * color_scale[None,None,:], 0, 255)
    return img, trans, inv_trans, rot, do_flip

def get_patch_img(img, bbox, scale, rot, do_flip, out_shape):
    img = img.copy()
    img_height, img_width, img_channels = img.shape
   
    bb_c_x = float(bbox[0] + 0.5*bbox[2])
    bb_c_y = float(bbox[1] + 0.5*bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1

    trans = get_affine_trans_mat(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot)
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
    img_patch = img_patch.astype(np.float32)
    inv_trans = get_affine_trans_mat(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot, inv=True)

    return img_patch, trans, inv_trans

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def get_affine_trans_mat(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir
    
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans

def process_kpt(kpt_img, kpt_cam, kpt_valid, do_flip, img_shape, flip_pair, img2bb_trans, rot, inp_kpt_name, tgt_kpt_name):
    kpt_img, kpt_cam, kpt_valid = kpt_img.copy(), kpt_cam.copy(), kpt_valid.copy()
    
    # flip augmentation
    if do_flip:
        kpt_cam[:,0] = -kpt_cam[:,0]
        kpt_img[:,0] = img_shape[1] - 1 - kpt_img[:,0]
        for pair in flip_pair:
            kpt_img[pair[0],:], kpt_img[pair[1],:] = kpt_img[pair[1],:].copy(), kpt_img[pair[0],:].copy()
            kpt_cam[pair[0],:], kpt_cam[pair[1],:] = kpt_cam[pair[1],:].copy(), kpt_cam[pair[0],:].copy()
            kpt_valid[pair[0],:], kpt_valid[pair[1],:] = kpt_valid[pair[1],:].copy(), kpt_valid[pair[0],:].copy()
    
    # 3D data rotation augmentation
    rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0], 
    [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
    [0, 0, 1]], dtype=np.float32)
    kpt_cam = np.dot(rot_aug_mat, kpt_cam.transpose(1,0)).transpose(1,0)

    # affine transformation
    kpt_img_xy1 = np.concatenate((kpt_img, np.ones_like(kpt_img[:,:1])),1)
    kpt_img = np.dot(img2bb_trans, kpt_img_xy1.transpose(1,0)).transpose(1,0)
    kpt_img[:,0] = kpt_img[:,0] / cfg.input_img_shape[1] * cfg.vit_output_shape[1]
    kpt_img[:,1] = kpt_img[:,1] / cfg.input_img_shape[0] * cfg.vit_output_shape[0]
    
    # check truncation
    kpt_trunc = kpt_valid * ((kpt_img[:,0] >= 0) * (kpt_img[:,0] < cfg.vit_output_shape[1]) * \
                (kpt_img[:,1] >= 0) * (kpt_img[:,1] < cfg.vit_output_shape[0])).reshape(-1,1).astype(np.float32)

    # change keypoint name
    if (inp_kpt_name is not None) and (tgt_kpt_name is not None):
        kpt_img = change_kpt_name(kpt_img, inp_kpt_name, tgt_kpt_name)
        kpt_cam = change_kpt_name(kpt_cam, inp_kpt_name, tgt_kpt_name)
        kpt_valid = change_kpt_name(kpt_valid, inp_kpt_name, tgt_kpt_name)
        kpt_trunc = change_kpt_name(kpt_trunc, inp_kpt_name, tgt_kpt_name)
    return kpt_img, kpt_cam, kpt_valid, kpt_trunc

def process_smplx_param(smplx_param, cam_param, do_flip, img_shape, img2bb_trans, rot):
    pose_valid = np.ones((smpl_x.joint['num'],1), dtype=np.float32)
    kpt_valid = np.ones((smpl_x.kpt['num'],1), dtype=np.float32)

    root_pose, body_pose, shape_param, trans = smplx_param['root_pose'], smplx_param['body_pose'], smplx_param['shape'], smplx_param['trans']
    if 'lhand_pose' in smplx_param and smplx_param['lhand_valid']:
        lhand_pose = smplx_param['lhand_pose']
    else:
        lhand_pose = np.zeros((3*len(smpl_x.joint['part_idx']['lhand'])), dtype=np.float32)
        pose_valid[smpl_x.joint['part_idx']['lhand']] = 0
        kpt_valid[smpl_x.kpt['part_idx']['lhand']] = 0
    if 'rhand_pose' in smplx_param and smplx_param['rhand_valid']:
        rhand_pose = smplx_param['rhand_pose']
    else:
        rhand_pose = np.zeros((3*len(smpl_x.joint['part_idx']['rhand'])), dtype=np.float32)
        pose_valid[smpl_x.joint['part_idx']['rhand']] = 0
        kpt_valid[smpl_x.kpt['part_idx']['rhand']] = 0
    if 'jaw_pose' in smplx_param and 'expr' in smplx_param and smplx_param['face_valid']:
        jaw_pose = smplx_param['jaw_pose']
        expr_param = smplx_param['expr']
        expr_valid = float(True)
    else:
        jaw_pose = np.zeros((3), dtype=np.float32)
        expr_param = np.zeros((smpl_x.expr_param_dim), dtype=np.float32)
        pose_valid[smpl_x.joint['part_idx']['face']] = 0
        kpt_valid[smpl_x.kpt['part_idx']['face']] = 0
        expr_valid = float(False)
    if 'leye_pose' in smplx_param and 'reye_pose' in smplx_param and smplx_param['face_valid']:
        leye_pose = smplx_param['leye_pose']
        reye_pose = smplx_param['reye_pose']
    else:
        leye_pose = torch.zeros((1,3)).float()
        reye_pose = torch.zeros((1,3)).float() 
        pose_valid[smpl_x.joint['name'].index('L_Eye')] = 0
        pose_valid[smpl_x.joint['name'].index('R_Eye')] = 0
        kpt_valid[smpl_x.kpt['name'].index('L_Eye')] = 0
        kpt_valid[smpl_x.kpt['name'].index('R_Eye')] = 0
    if 'gender' in smplx_param:
        gender = smplx_param['gender']
    else:
        gender = 'neutral'
    root_pose = torch.FloatTensor(root_pose).view(1,3) # (1,3)
    body_pose = torch.FloatTensor(body_pose).view(-1,3) # (21,3)
    lhand_pose = torch.FloatTensor(lhand_pose).view(-1,3) # (15,3)
    rhand_pose = torch.FloatTensor(rhand_pose).view(-1,3) # (15,3)
    jaw_pose = torch.FloatTensor(jaw_pose).view(-1,3) # (1,3)
    leye_pose = torch.FloatTensor(leye_pose).view(-1,3) # (1,3)
    reye_pose = torch.FloatTensor(reye_pose).view(-1,3) # (1,3)
    shape_param = torch.FloatTensor(shape_param).view(1,-1) # shape parameter
    expr_param = torch.FloatTensor(expr_param).view(1,-1) # expression parameter
    trans = torch.FloatTensor(trans).view(1,-1) # translation vector

    # apply camera extrinsic (rotation)
    # merge root pose and camera rotation 
    if 'R' in cam_param:
        R = torch.FloatTensor(cam_param['R']).view(3,3)
        root_pose = axis_angle_to_matrix(root_pose).view(3,3)
        root_pose = matrix_to_axis_angle(torch.matmul(R, root_pose)).view(1,3)

    # get vertex and keypoint coordinates
    with torch.no_grad():
        output = smpl_x.layer[gender](global_orient=root_pose, body_pose=body_pose.view(1,-1), left_hand_pose=lhand_pose.view(1,-1), right_hand_pose=rhand_pose.view(1,-1), jaw_pose=jaw_pose.view(1,-1), leye_pose=leye_pose, reye_pose=reye_pose, expression=expr_param, betas=shape_param, transl=trans)
    vert_cam = output.vertices[0].numpy()
    kpt_cam = output.joints[0].numpy()[smpl_x.kpt['idx'],:]

    # apply camera exrinsic (translation)
    # compenstate rotation (translation from origin to root joint was not cancled)
    if 'R' in cam_param and 't' in cam_param:
        R, t = np.array(cam_param['R'], dtype=np.float32).reshape(3,3), np.array(cam_param['t'], dtype=np.float32).reshape(1,3)
        root_cam = kpt_cam[smpl_x.kpt['root_idx'],None,:]
        kpt_cam = kpt_cam - root_cam + np.dot(R, root_cam.transpose(1,0)).transpose(1,0) + t
        vert_cam = vert_cam - root_cam + np.dot(R, root_cam.transpose(1,0)).transpose(1,0) + t
    vert_cam_wo_aug = vert_cam # do not apply data augmentation to vertices

    # keypoint
    kpt_img = cam2img(kpt_cam, cam_param['focal'], cam_param['princpt'])[:,:2]
    kpt_cam = kpt_cam - kpt_cam[smpl_x.kpt['root_idx'],None,:] # root-relative
    kpt_img, kpt_cam, kpt_valid, kpt_trunc = process_kpt(kpt_img, kpt_cam, kpt_valid, do_flip, img_shape, smpl_x.kpt['flip_pair'], img2bb_trans, rot, None, None)

    # root pose
    rot = torch.FloatTensor([float(rot)])
    rot_aug_mat = torch.FloatTensor([[torch.cos(torch.deg2rad(-rot)), -torch.sin(torch.deg2rad(-rot)), 0], 
                                    [torch.sin(torch.deg2rad(-rot)), torch.cos(torch.deg2rad(-rot)), 0],
                                    [0, 0, 1]])
    root_pose = axis_angle_to_matrix(root_pose).view(3,3)
    root_pose = matrix_to_axis_angle(torch.matmul(rot_aug_mat, root_pose)).view(1,3)

    # pose
    pose = torch.cat((root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, leye_pose, reye_pose)).numpy() # follow smpl_x.joint['name']
    if do_flip:
        for pair in smpl_x.joint['flip_pair']:
            pose[pair[0], :], pose[pair[1], :] = pose[pair[1], :].copy(), pose[pair[0], :].copy()
            pose_valid[pair[0],:], pose_valid[pair[1],:] = pose_valid[pair[1],:].copy(), pose_valid[pair[0],:].copy()
        pose[:,1:3] *= -1 # multiply -1 to y and z axis of axis-angle
    
    shape_param = shape_param.numpy().reshape(-1)
    expr_param = expr_param.numpy().reshape(-1)
    return pose, shape_param, expr_param, pose_valid, expr_valid, kpt_cam, kpt_img, kpt_valid, kpt_trunc, vert_cam_wo_aug
    
def process_mano_param(mano_param, cam_param, do_flip, img_shape, rot):
    pose, shape_param, trans = mano_param['pose'], mano_param['shape'], mano_param['trans']
    hand_type = mano_param['hand_type']
    pose = torch.FloatTensor(pose).view(-1,3); shape_param = torch.FloatTensor(shape_param).view(1,-1); # mano parameters (pose: 48 dimension, shape: 10 dimension)
    trans = torch.FloatTensor(trans).view(1,-1) # translation vector
    
    # apply camera extrinsic (rotation)
    # merge root pose and camera rotation
    if 'R' in cam_param:
        R = np.array(cam_param['R'], dtype=np.float32).reshape(3,3)
        root_pose = pose[mano.joint['root_idx'],:].numpy()
        root_pose, _ = cv2.Rodrigues(root_pose)
        root_pose, _ = cv2.Rodrigues(np.dot(R,root_pose))
        pose[mano.joint['root_idx']] = torch.from_numpy(root_pose).view(3)

    # flip pose parameter (axis-angle)
    if do_flip:
        if hand_type == 'right':
            hand_type = 'left'
        else:
            hand_type = 'right'
        pose[:,1:3] *= -1 # multiply -1 to y and z axis of axis-angle
        trans[:,0] *= -1 # multiply -1

    # get vertex and keypoint coordinates
    root_pose = pose[mano.joint['root_idx']].view(1,3)
    hand_pose = torch.cat((pose[:mano.joint['root_idx'],:], pose[mano.joint['root_idx']+1:,:])).view(1,-1)
    with torch.no_grad():
        output = mano.layer[hand_type](global_orient=root_pose, hand_pose=hand_pose, betas=shape_param, transl=trans)
    vert_cam = output.vertices[0].numpy()
    kpt_cam = np.dot(mano.kpt['regressor'], vert_cam)

    # bring geometry to the original (before flip) position
    if do_flip:
        flip_trans_x = kpt_cam[mano.kpt['root_idx'],0] * -2
        vert_cam[:,0] += flip_trans_x
        kpt_cam[:,0] += flip_trans_x
    
    # apply camera exrinsic (translation)
    # compenstate rotation (translation from origin to root joint was not cancled)
    if 'R' in cam_param and 't' in cam_param:
        R, t = np.array(cam_param['R'], dtype=np.float32).reshape(3,3), np.array(cam_param['t'], dtype=np.float32).reshape(1,3)
        root_cam = kpt_cam[mano.kpt['root_idx'],None,:]
        kpt_cam = kpt_cam - root_cam + np.dot(R, root_cam.transpose(1,0)).transpose(1,0) + t
        vert_cam = vert_cam - root_cam + np.dot(R, root_cam.transpose(1,0)).transpose(1,0) + t

    # flip translation
    if do_flip: # avg of old and new root joint should be image center.
        focal, princpt = cam_param['focal'], cam_param['princpt']
        flip_trans_x = 2 * (((img_shape[1] - 1)/2. - princpt[0]) / focal[0] * kpt_cam[mano.kpt['root_idx'],2]) - 2 * kpt_cam[mano.kpt['root_idx'],0]
        vert_cam[:,0] += flip_trans_x
        kpt_cam[:,0] += flip_trans_x

    # project keypoint to the image space
    if 'D' in cam_param:
        kpt_img = distort_projection_fisheye(torch.from_numpy(kpt_cam)[None], torch.from_numpy(cam_param['focal'])[None], torch.from_numpy(cam_param['princpt'])[None], torch.from_numpy(cam_param['D'])[None])
        kpt_img = kpt_img[0].numpy()[:,:2]
    else:
        kpt_img = cam2img(kpt_cam, cam_param['focal'], cam_param['princpt'])[:,:2]

    # root pose
    rot = torch.FloatTensor([float(rot)])
    rot_aug_mat = torch.FloatTensor([[torch.cos(torch.deg2rad(-rot)), -torch.sin(torch.deg2rad(-rot)), 0], 
                                    [torch.sin(torch.deg2rad(-rot)), torch.cos(torch.deg2rad(-rot)), 0],
                                    [0, 0, 1]])
    root_pose = axis_angle_to_matrix(root_pose).view(3,3)
    root_pose = matrix_to_axis_angle(torch.matmul(rot_aug_mat, root_pose)).view(3)

    root_pose = root_pose.view(3).numpy()
    hand_pose = hand_pose.view(-1,3).numpy() 
    shape_param = shape_param.view(-1).numpy()
    return root_pose, hand_pose, shape_param, kpt_cam, kpt_img, vert_cam

