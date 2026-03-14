import json
import numpy as np
from glob import glob
import os.path as osp

root_path = '/data/ReInterHand'
capture_path_list = [x for x in glob(osp.join(root_path, '*')) if osp.isdir(x)]
for capture_path in capture_path_list:
    cam_path = osp.join(capture_path, 'Mugsy_cameras', 'cam_params.json')
    with open(cam_path) as f:
        cam_params = json.load(f)
    R = np.stack([np.array(cam_params[cam_id]['R'], dtype=np.float32) for cam_id in cam_params.keys()]).reshape(-1,3,3)

    # camera coordinate -> world coordinate
    cam_axis = np.array([[1,0,0], [0,-1,0], [0,0,1]], dtype=np.float32) # (0,1,0) in the camera coordinate system represents downward direction. multiply -1 to make it upward.
    cam_axis = np.stack([np.dot(R[i].transpose(1,0), cam_axis.transpose(1,0)).transpose(1,0) for i in range(len(R))]).reshape(-1,3,3)
    up_vec = cam_axis.mean(0)[1]
    print(up_vec)


