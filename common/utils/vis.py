import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
PointLights,
PerspectiveCameras,
Materials,
SoftPhongShader,
RasterizationSettings,
MeshRendererWithFragments,
MeshRasterizer,
TexturesVertex)
os.environ["PYOPENGL_PLATFORM"] = "egl"

def vis_kpt(img, kpt):
    img = img.copy()
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kpt) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
    for i in range(len(kpt)):
        p = kpt[i][0].astype(np.int32), kpt[i][1].astype(np.int32)
        cv2.circle(img, p, radius=3, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)
    return img


def render_mesh(mesh, face, cam_param, bkg, blend_ratio=1.0, color=None):
    mesh = torch.FloatTensor(mesh).cuda()[None,:,:]
    face = torch.LongTensor(face.astype(np.int64)).cuda()[None,:,:]
    cam_param = {k: torch.FloatTensor(v).cuda()[None,:] for k,v in cam_param.items()}
    render_shape = (bkg.shape[0], bkg.shape[1]) # height, width

    batch_size, vertex_num = mesh.shape[:2]
    if color is None:
        textures = TexturesVertex(verts_features=torch.ones((batch_size,vertex_num,3)).float().cuda())
    else:
        #textures = torch.FloatTensor(color).float().cuda()[None,None,:].repeat(batch_size,vertex_num,1)
        #textures = TexturesVertex(verts_features=textures)
        textures = TexturesVertex(verts_features=color)
    mesh = torch.stack((-mesh[:,:,0], -mesh[:,:,1], mesh[:,:,2]),2) # reverse x- and y-axis following PyTorch3D axis direction
    mesh = Meshes(mesh, face, textures)

    cameras = PerspectiveCameras(focal_length=cam_param['focal'],
                                principal_point=cam_param['princpt'],
                                device='cuda',
                                in_ndc=False,
                                image_size=torch.LongTensor(render_shape).cuda().view(1,2))
    raster_settings = RasterizationSettings(image_size=render_shape, blur_radius=0.0, faces_per_pixel=1, max_faces_per_bin=20000)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).cuda()
    lights = PointLights(device='cuda')
    shader = SoftPhongShader(device='cuda', cameras=cameras, lights=lights)
    materials = Materials(
	device='cuda',
	specular_color=[[0.0, 0.0, 0.0]],
	shininess=0.0
    )

    # render
    with torch.no_grad():
        renderer = MeshRendererWithFragments(rasterizer=rasterizer, shader=shader)
        images, fragments = renderer(mesh, materials=materials)

    # background masking
    is_bkg = (fragments.zbuf <= 0).float().cpu().numpy()[0]
    render = images[0,:,:,:3].cpu().numpy()
    fg = render * blend_ratio + bkg/255 * (1 - blend_ratio)
    render = fg * (1 - is_bkg) * 255 + bkg * is_bkg
    return render
