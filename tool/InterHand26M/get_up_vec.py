import json
import numpy as np
from glob import glob
import os.path as osp

root_path = '/data/InterHand26M/5fps/annotations'
for split in ('train', 'test', 'val'):
    cam_path = osp.join(root_path, split, 'InterHand2.6M_' + split + '_camera.json')
    with open(cam_path) as f:
        cam_params = json.load(f)
    for capture_id in cam_params.keys():
        R = np.stack([np.array(v, dtype=np.float32) for k,v in cam_params[capture_id]['camrot'].items()]).reshape(-1,3,3)
        
        # camera coordinate -> world coordinate
        cam_axis = np.array([[1,0,0], [0,-1,0], [0,0,1]], dtype=np.float32) # (0,1,0) in the camera coordinate system represents downward direction. multiply -1 to make it upward.
        cam_axis = np.stack([np.dot(R[i].transpose(1,0), cam_axis.transpose(1,0)).transpose(1,0) for i in range(len(R))]).reshape(-1,3,3)
        up_vec = cam_axis.mean(0)[1]
        print(up_vec)

