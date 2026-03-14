import torch
import numpy as np
from torch.nn import functional as F
import torchgeometry as tgm
from pytorch3d.transforms import matrix_to_axis_angle
from pytorch3d.ops import corresponding_points_alignment

def cam2img(cam_coord, f, c):
    x = cam_coord[:,0] / cam_coord[:,2] * f[0] + c[0]
    y = cam_coord[:,1] / cam_coord[:,2] * f[1] + c[1]
    z = cam_coord[:,2]
    return np.stack((x,y,z),1)

def img2cam(img_coord, f, c):
    x = (img_coord[:,0] - c[0]) / f[0] * img_coord[:,2]
    y = (img_coord[:,1] - c[1]) / f[1] * img_coord[:,2]
    z = img_coord[:,2]
    return np.stack((x,y,z),1)

def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord

def cam2world(cam_coord, R, t):
    world_coord = np.dot(np.linalg.inv(R), (cam_coord - t.reshape(1,3)).transpose(1,0)).transpose(1,0)
    return world_coord

def change_kpt_name(inp_kpt, inp_name, tgt_name):
    inp_kpt_num = len(inp_name)
    tgt_kpt_num = len(tgt_name)

    new_kpt = np.zeros(((tgt_kpt_num,) + inp_kpt.shape[1:]), dtype=np.float32)
    for inp_idx in range(len(inp_name)):
        name = inp_name[inp_idx]
        if name in tgt_name:
            tgt_idx = tgt_name.index(name)
            new_kpt[tgt_idx] = inp_kpt[inp_idx]

    return new_kpt

def distort_projection_fisheye(point, focal, princpt, D):
    z = point[:,:,2].clone()

    # distort
    point_ndc = point[:,:,:2] / z[:,:,None]
    r = torch.sqrt(torch.sum(point_ndc ** 2, 2))
    theta = torch.atan(r)
    theta_d = theta * (
            1
            + D[:,None,0] * theta.pow(2)
            + D[:,None,1] * theta.pow(4)
            + D[:,None,2] * theta.pow(6)
            + D[:,None,3] * theta.pow(8)
    )
    point_ndc = point_ndc * (theta_d / r)[:,:,None]

    # project
    x = point_ndc[:,:,0]
    y = point_ndc[:,:,1]
    x = x * focal[:,None,0] + princpt[:,None,0]
    y = y * focal[:,None,1] + princpt[:,None,1]
    point_proj = torch.stack((x,y,z),2)
    return point_proj

def rotation_6d_to_axis_angle(x):
    batch_size = x.shape[0]
    
    # rotation_6d -> matrix
    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    rot_mat = torch.stack((b1, b2, b3), dim=-1) # this is different from pytorch3d.transforms.rotation_6d_to_matrix as its stack dimension is -2. 
    
    # matrix -> axis_angle
    axis_angle = matrix_to_axis_angle(rot_mat)
    return axis_angle

def soft_argmax_3d(heatmap3d):
    batch_size = heatmap3d.shape[0]
    depth, height, width = heatmap3d.shape[2:]
    heatmap3d = heatmap3d.reshape((batch_size, -1, depth * height * width))
    heatmap3d = F.softmax(heatmap3d, 2)
    heatmap3d = heatmap3d.reshape((batch_size, -1, depth, height, width))

    accu_x = heatmap3d.sum(dim=(2, 3))
    accu_y = heatmap3d.sum(dim=(2, 4))
    accu_z = heatmap3d.sum(dim=(3, 4))

    accu_x = accu_x * torch.arange(width).float().cuda()[None, None, :]
    accu_y = accu_y * torch.arange(height).float().cuda()[None, None, :]
    accu_z = accu_z * torch.arange(depth).float().cuda()[None, None, :]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    accu_z = accu_z.sum(dim=2, keepdim=True)

    coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)
    return coord_out

def restore_bbox(bbox_center, bbox_size, aspect_ratio, extension_ratio):
    bbox = bbox_center.view(-1, 1, 2) + torch.cat((-bbox_size.view(-1, 1, 2) / 2., bbox_size.view(-1, 1, 2) / 2.),1)
    bbox = bbox.view(-1, 4)

    # xyxy -> xywh
    bbox[:, 2] = bbox[:, 2] - bbox[:, 0]
    bbox[:, 3] = bbox[:, 3] - bbox[:, 1]

    # aspect ratio preserving bbox
    w = bbox[:, 2]
    h = bbox[:, 3]
    c_x = bbox[:, 0] + w / 2.
    c_y = bbox[:, 1] + h / 2.

    mask1 = w > (aspect_ratio * h)
    mask2 = w < (aspect_ratio * h)
    h[mask1] = w[mask1] / aspect_ratio
    w[mask2] = h[mask2] * aspect_ratio

    bbox[:, 2] = w * extension_ratio
    bbox[:, 3] = h * extension_ratio
    bbox[:, 0] = c_x - bbox[:, 2] / 2.
    bbox[:, 1] = c_y - bbox[:, 3] / 2.

    # xywh -> xyxy
    bbox[:, 2] = bbox[:, 2] + bbox[:, 0]
    bbox[:, 3] = bbox[:, 3] + bbox[:, 1]
    return bbox

def rigid_transform_3D(A, B):
    n, dim = A.shape
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B) / n
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))

    varP = np.var(A, axis=0).sum()
    c = 1 / varP * np.sum(s)

    t = -np.dot(c * R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return c, R, t


def rigid_align(A, B):
    c, R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(c * R, np.transpose(A))) + t
    return A2
