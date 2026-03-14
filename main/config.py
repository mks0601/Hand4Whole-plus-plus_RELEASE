import os
import os.path as osp
import sys

class Config:
    
    ## dataset (use names in the `data` folder)
    trainset = ['InterHand26M', 'ReInterHand', 'ARCTIC'] 
    trainset_sample_prob = [0.4, 0.3, 0.3]
    #trainset = ['InterHand26M', 'ReInterHand', 'ARCTIC', 'AGORA'] 
    #trainset_sample_prob = [0.25, 0.25, 0.25, 0.25]
    testset = 'ARCTIC'

    ## input, output
    input_img_shape = (512, 384)
    input_body_shape = (256, 192)
    input_hand_shape = (256, 256)
    input_face_shape = (192, 192)
    vit_output_shape = (16, 12)
    output_hm_shape = (16, 16, 12)
    vit_feat_dim = 1024
    focal = (5000, 5000) # virtual focal lengths
    princpt = (input_body_shape[1]/2, input_body_shape[0]/2) # virtual principal point position
    body_3d_size = 2
    bbox_3d_size = 2.5

    ## training config
    lr = 1e-4
    lr_dec_factor = 10
    lr_dec_epoch = [4,6] 
    end_epoch = 7 
    train_batch_size = 32

    ## testing config
    test_batch_size = 64

    ## others
    num_thread = 16
    num_gpus = 1
    continue_train = False
    
    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    model_dir = osp.join(output_dir, 'model_dump')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')
    human_model_path = osp.join(root_dir, 'common', 'utils', 'human_model_files')
    wilor_root_path = osp.join(root_dir, 'common', 'nets', 'WiLoR')
    dwpose_path = {'root': osp.join(root_dir, 'common', 'nets', 'mmpose'),
                'cfg': osp.join(root_dir, 'common', 'nets', 'mmpose', 'configs/wholebody_2d_keypoint/rtmpose/ubody/rtmpose-l_8xb64-270e_coco-ubody-wholebody-256x192.py'),
                'ckpt': osp.join(root_dir, 'common', 'nets', 'mmpose', 'dw-ll_ucoco.pth')}

    def set_args(self, continue_train=False):
        self.continue_train = continue_train

cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
sys.path.insert(0, cfg.wilor_root_path)
sys.path.insert(0, cfg.dwpose_path['root'])
from utils.dir import add_pypath, make_folder
add_pypath(osp.join(cfg.data_dir))
for i in range(len(cfg.trainset)):
    add_pypath(osp.join(cfg.data_dir, cfg.trainset[i]))
add_pypath(osp.join(cfg.data_dir, cfg.testset))
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)
