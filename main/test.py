import torch
import argparse
from tqdm import tqdm
from config import cfg
from base import Tester
from utils.vis import vis_kpt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    args = parser.parse_args()

    assert args.test_epoch, 'Test epoch is required.'
    return args

def main():

    args = parse_args()
    cfg.set_args()

    tester = Tester(args.test_epoch)
    tester._make_batch_generator()
    tester._make_model()
    
    eval_result = {}
    cur_sample_idx = 0
    for itr, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator)):
        
        # forward
        with torch.no_grad():
            out = tester.model(inputs, targets, meta_info, 'test')
        
        """
        # for debug
        import cv2
        from utils.vis import render_mesh
        from pytorch3d.io import save_obj
        from utils.smpl_x import smpl_x
        from utils.mano import mano
        import numpy as np
        rhand_color = [0.6, 0.7, 1.0]
        lhand_color = [0.7, 1.0, 0.7]
        color = torch.ones((1,smpl_x.vertex_num,3)).float().cuda()
        color[:,smpl_x.hand_vertex_idx['right_hand'],:] = torch.FloatTensor(rhand_color).cuda()[None,:]
        color[:,smpl_x.hand_vertex_idx['left_hand'],:] = torch.FloatTensor(lhand_color).cuda()[None,:]
        filename = str(cur_sample_idx)
        img = inputs['img'][0].cpu().numpy().transpose(1,2,0)*1.3*255
        focal = (cfg.focal[0]/cfg.input_body_shape[1]*cfg.input_img_shape[1], cfg.focal[1]/cfg.input_body_shape[0]*cfg.input_img_shape[0])
        princpt = (cfg.princpt[0]/cfg.input_body_shape[1]*cfg.input_img_shape[1], cfg.princpt[1]/cfg.input_body_shape[0]*cfg.input_img_shape[0])
        cv2.imwrite(filename + '_input.jpg', img[:,:,::-1])
        render = render_mesh(out['smplx_vert_cam'][0].cpu().numpy(), smpl_x.face, {'focal': focal, 'princpt': princpt}, img, color=color)
        cv2.imwrite(filename + '.jpg', render[:,:,::-1])
        #render = render_mesh(out['rmano_vert_cam'][0].cpu().numpy(), mano.face['right'], {'focal': focal, 'princpt': princpt}, img)
        #cv2.imwrite(filename + '_rmano.jpg', render)
        #render = render_mesh(out['lmano_vert_cam'][0].cpu().numpy(), mano.face['left'], {'focal': focal, 'princpt': princpt}, img)
        #cv2.imwrite(filename + '_lmano.jpg', render)
        save_obj(filename + '.obj', out['smplx_vert_cam'][0].cpu(), torch.LongTensor(smpl_x.face))
        """

        # save output
        out = {k: v.cpu().numpy() for k,v in out.items()}
        for k,v in out.items(): batch_size = out[k].shape[0]
        out = [{k: v[bid] for k,v in out.items()} for bid in range(batch_size)]
        
        # evaluate
        cur_eval_result = tester._evaluate(out, cur_sample_idx)
        for k,v in cur_eval_result.items():
            if k in eval_result: eval_result[k] += v
            else: eval_result[k] = v
        cur_sample_idx += len(out)
    
    tester._print_eval_result(eval_result)

if __name__ == "__main__":
    main()
