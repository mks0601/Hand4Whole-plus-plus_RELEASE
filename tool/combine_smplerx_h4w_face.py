import torch

smpler_x = torch.load('smpler_x_l32.pth.tar')
smpler_x['epoch'] = 0

h4w = torch.load('snapshot_6.pth.tar')
for k,v in h4w['network'].items():
    if ('face_roi_net' in k) or ('face_regressor' in k):
        smpler_x['network'][k] = v

torch.save(smpler_x, 'snapshot_0.pth')
