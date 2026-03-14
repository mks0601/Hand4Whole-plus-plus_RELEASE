# AGORA dataset provie gendered shape parameters
# change them to gender-neutral parameters

import torch
import torch.nn as nn
import numpy as np
import json
import os.path as osp
from pytorch3d.io import load_obj, save_obj
import smplx
import torch.optim
from glob import glob
from tqdm import tqdm
from pycocotools.coco import COCO
from torch.utils.data import DataLoader

class AGORA(torch.utils.data.Dataset):
    def __init__(self, root_path):
        self.root_path = root_path
        self.datalist = self.load_data()
 
    def load_data(self):
        datalist = []
        for split in ('train', 'validation'):
            db = COCO(osp.join(self.root_path, 'AGORA_' + split + '_SMPLX.json'))
            
            for aid in db.anns.keys():
                ann = db.anns[aid]
                image_id = ann['image_id']
                img = db.loadImgs(image_id)[0]
                person_id = ann['person_id']
                if not ann['is_valid']:
                    continue
                
                smplx_param_path = osp.join(self.root_path, ann['smplx_param_path'])
                data_dict = {'smplx_param_path': smplx_param_path, 'gender': ann['gender']}
                datalist.append(data_dict)

        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = self.datalist[idx]
        with open(data['smplx_param_path']) as f:
            shape_param = torch.FloatTensor(json.load(f)['betas']).view(-1)
        is_male = (data['gender'] == 'male')
        save_path = data['smplx_param_path'].replace('.json', '_neutral_gender_betas.json')
        return {'shape_param': shape_param, 'is_male': is_male, 'save_path': save_path}

root_path = '/data/AGORA'
dataset_loader = AGORA(root_path)
batch_generator = DataLoader(dataset=dataset_loader, batch_size=128, shuffle=False, num_workers=8)
smplx_layer = {k: smplx.create('/home/mks0601/workspace/human_model_files', 'smplx', gender=k, use_pca=False, flat_hand_mean=True).cuda() for k in ['male', 'female', 'neutral']}
for data in tqdm(batch_generator):
    # get gendered target
    with torch.no_grad():
        shape_param = data['shape_param'].cuda()
        is_male = data['is_male'].view(-1,1,1).float().cuda()
        batch_size = shape_param.shape[0]
        zero_pose = torch.zeros((batch_size,3)).float().cuda()
        zero_body_pose = torch.zeros((batch_size,63)).float().cuda()
        zero_hand_pose = torch.zeros((batch_size,45)).float().cuda()
        zero_expr = torch.zeros((batch_size,10)).float().cuda()
        vert_tgt = smplx_layer['male'](betas=shape_param, global_orient=zero_pose, body_pose=zero_body_pose, left_hand_pose=zero_hand_pose, right_hand_pose=zero_hand_pose, jaw_pose=zero_pose, leye_pose=zero_pose, reye_pose=zero_pose, expression=zero_expr).vertices*is_male + \
                    smplx_layer['female'](betas=shape_param, global_orient=zero_pose, body_pose=zero_body_pose, left_hand_pose=zero_hand_pose, right_hand_pose=zero_hand_pose, jaw_pose=zero_pose, leye_pose=zero_pose, reye_pose=zero_pose, expression=zero_expr).vertices*(1-is_male)
        vert_tgt = vert_tgt.detach()
    
    # optimize gender-neutral shape parameter
    shape_param = nn.Parameter(torch.zeros((batch_size,10)).float().cuda())
    optimizer = torch.optim.Adam([shape_param], lr=1e-2)
    for itr in range(500):
        if itr == 350:
            for g in optimizer.param_groups:
                g['lr'] = 1e-3

        # forwrad
        optimizer.zero_grad()
        output = smplx_layer['neutral'](betas=shape_param, global_orient=zero_pose, body_pose=zero_body_pose, left_hand_pose=zero_hand_pose, right_hand_pose=zero_hand_pose, jaw_pose=zero_pose, leye_pose=zero_pose, reye_pose=zero_pose, expression=zero_expr)
        vert_out = output.vertices
        loss = torch.abs(vert_out - vert_tgt).mean()
        
        # backward
        loss.backward()
        optimizer.step()

        #print(itr, loss)
    
    for i in range(batch_size):
        save_path = data['save_path'][i]
        with open(save_path, 'w') as f:
            json.dump(shape_param[i].detach().cpu().numpy().tolist(), f)


