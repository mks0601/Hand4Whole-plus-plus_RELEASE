import sys
import os
import os.path as osp
import numpy as np
import cv2
import json
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn

sys.path.insert(0, osp.join('..', 'main'))
from config import cfg
from model import get_model
from utils.preprocessing import load_img, set_aspect_ratio, get_patch_img
from utils.smpl_x import smpl_x
from utils.vis import render_mesh
from ultralytics import YOLO
from pytorch3d.io import save_obj
from glob import glob
from tqdm import tqdm

def get_one_box(boxes):
    best_person_box = None
    max_conf = -1
    for box in boxes:
        if int(box.cls) == 0:  # person class
            conf = float(box.conf)  # confidence score
            if conf > max_conf:
                max_conf = conf
                best_person_box = box.xyxy[0].tolist()
    bbox = best_person_box
    bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]] # xywh
    return bbox

root_path = osp.abspath('.')
input_root_path = osp.join(root_path, 'inputs')
output_root_path = osp.join(root_path, 'outputs')
os.makedirs(output_root_path, exist_ok=True)
rhand_color = [0.6, 0.7, 1.0]
lhand_color = [0.7, 1.0, 0.7]


# snapshot load
model_path = osp.join(root_path, 'snapshot_6.pth')
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))
model = get_model('test')
model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
for module in model.module.trainable_modules+model.module.eval_modules:
    module.eval()
cudnn.benchmark = True

# human detector
detector = YOLO("yolo11n.pt")

# prepare input image
transform = transforms.ToTensor()
img_path_list = glob(osp.join(input_root_path, '*'))
for img_path in tqdm(img_path_list):
    file_name = img_path.split('/')[-1][:-4]
    original_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    original_img_height, original_img_width = original_img.shape[:2]

    # prepare bbox
    results = detector(img_path, verbose=False)
    boxes = results[0].boxes
    bbox = get_one_box(boxes)
    #bbox = [0, 0, original_img_width, original_img_height]

    bbox = set_aspect_ratio(bbox, cfg.input_img_shape[1]/cfg.input_img_shape[0])
    img, img2bb_trans, bb2img_trans = get_patch_img(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape) 
    img = transform(img.astype(np.float32))/255
    img = img.cuda()[None,:,:,:]

    # forward
    inputs = {'img': img}
    targets = {}
    meta_info = {}
    with torch.no_grad():
        out = model(inputs, targets, meta_info, 'test')
    vert = out['smplx_vert_cam'].detach().cpu().numpy()[0]

    # save mesh
    #save_obj(osp.join(output_root_path, file_name + '.obj'), torch.FloatTensor(vert), torch.LongTensor(smpl_x.face))
    color = np.ones((smpl_x.vertex_num,3), dtype=np.float32) * 0.8
    color[smpl_x.hand_vertex_idx['right_hand'],:] = np.array(rhand_color, dtype=np.float32).reshape(1,3) * 0.8
    color[smpl_x.hand_vertex_idx['left_hand'],:] = np.array(lhand_color, dtype=np.float32).reshape(1,3) * 0.8
    def save_obj_w_color(v, f, color=None, file_name='output.obj'):
        obj_file = open(file_name, 'w')
        for i in range(len(v)):
            if color is None:
                obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
            else:
                obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + ' ' + str(color[i][0]) + ' ' + str(color[i][1]) + ' ' + str(color[i][2]) + '\n')
        for i in range(len(f)):
            obj_file.write('f ' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '\n')
        obj_file.close()
    save_obj_w_color(vert, smpl_x.face, color, osp.join(output_root_path, file_name + '.obj'))

    # render mesh (cropped image space)
    vis_img = img.cpu().numpy()[0].transpose(1,2,0).copy() * 255
    focal = [cfg.focal[0] / cfg.input_body_shape[1] * cfg.input_img_shape[1], cfg.focal[1] / cfg.input_body_shape[0] * cfg.input_img_shape[0]]
    princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * cfg.input_img_shape[1], cfg.princpt[1] / cfg.input_body_shape[0] * cfg.input_img_shape[0]]
    rendered_img = render_mesh(vert, smpl_x.face, {'focal': focal, 'princpt': princpt}, vis_img)[:,:,::-1]
    cv2.imwrite(osp.join(output_root_path, file_name + '_render_cropped_img.jpg'), rendered_img)

    # render mesh (original image space)
    vis_img = original_img.copy()
    focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
    princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]
    color = torch.ones((1,smpl_x.vertex_num,3)).float().cuda()
    color[:,smpl_x.hand_vertex_idx['right_hand'],:] = torch.FloatTensor(rhand_color).cuda()[None,:]
    color[:,smpl_x.hand_vertex_idx['left_hand'],:] = torch.FloatTensor(lhand_color).cuda()[None,:]
    rendered_img = render_mesh(vert, smpl_x.face, {'focal': focal, 'princpt': princpt}, vis_img, color=color)[:,:,::-1]
    cv2.imwrite(osp.join(output_root_path, file_name + '_render_original_img.jpg'), rendered_img)

    # save SMPL-X parameters
    root_pose = out['smplx_root_pose'].detach().cpu().numpy()[0]
    body_pose = out['smplx_body_pose'].detach().cpu().numpy()[0] 
    lhand_pose = out['smplx_lhand_pose'].detach().cpu().numpy()[0] 
    rhand_pose = out['smplx_rhand_pose'].detach().cpu().numpy()[0] 
    jaw_pose = out['smplx_jaw_pose'].detach().cpu().numpy()[0] 
    shape = out['smplx_shape'].detach().cpu().numpy()[0]
    expr = out['smplx_expr'].detach().cpu().numpy()[0] 
    with open(osp.join(output_root_path, file_name + '_smplx_param.json'), 'w') as f:
        json.dump({'root_pose': root_pose.reshape(-1).tolist(), \
                'body_pose': body_pose.reshape(-1).tolist(), \
                'lhand_pose': lhand_pose.reshape(-1).tolist(), \
                'rhand_pose': rhand_pose.reshape(-1).tolist(), \
                'jaw_pose': jaw_pose.reshape(-1).tolist(), \
                'shape': shape.reshape(-1).tolist(), \
                'expr': expr.reshape(-1).tolist()}, f)



