import os
import os.path as osp
from glob import glob
from tqdm import tqdm
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader

class ARCTIC(torch.utils.data.Dataset):
    def __init__(self, root_path):
        self.root_path = root_path
        self.datalist = self.load_data()
 
    def load_data(self):
        datalist = []
        
        subject_path_list = glob(osp.join(root_path, 'images', 's*'))
        for subject_path in subject_path_list:
            subject_name = subject_path.split('/')[-1] # s01

            seq_path_list = glob(osp.join(subject_path, '*'))
            for seq_path in seq_path_list:
                seq_name = seq_path.split('/')[-1]

                cam_path_list = [x for x in glob(osp.join(self.root_path, 'images', subject_name, seq_name, '*')) if x.split('/')[-1] != 'data']
                for cam_path in cam_path_list:
                    cam_name = cam_path.split('/')[-1]

                    img_path_list = glob(osp.join(cam_path, '*.jpg'))
                    for img_path in img_path_list:
                        frame_idx = int(img_path.split('/')[-1][:-4])
                        data_dict = {'img_path': img_path, 'subject_name': subject_name, 'seq_name': seq_name, 'cam_name': cam_name, 'frame_idx': frame_idx}
                        datalist.append(data_dict)

        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = self.datalist[idx]
        img_path, subject_name, seq_name, cam_name, frame_idx = data['img_path'], data['subject_name'], data['seq_name'], data['cam_name'], data['frame_idx']

        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]
        img = cv2.resize(img, (img_width//2, img_height//2))

        save_path = osp.join(self.root_path, 'images_1k', subject_name, seq_name, cam_name)
        os.makedirs(save_path, exist_ok=True)
        cv2.imwrite(osp.join(save_path, '%05d.jpg' % frame_idx), img)
        return True

root_path = '/data/ARCTIC/arctic/unpack/arctic_data/data'
dataset = ARCTIC(root_path)
batch_generator = DataLoader(dataset=dataset, batch_size=32, shuffle=False, num_workers=16, pin_memory=True, drop_last=False)
for _ in tqdm(batch_generator):
    pass

