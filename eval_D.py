from torch.nn import functional as F
from models import RCF
import cv2
import torch
import numpy as np
import imageio as Im

img_path = 'Titanic1.gif'
Img = Im.mimread(img_path)
model_path = './pretrained/checkpoint_epoch19.pth'
model = RCF()
if torch.cuda.is_available():
    state = torch.load(model_path)
else:
    state = torch.load(model_path, map_location='cpu')
state = state['state_dict']
model_dict = model.state_dict()  #去除上采样部位数据
state_dict = {k: v for k, v in state.items() if k in model_dict.keys()}
model_dict.update(state_dict)
model.load_state_dict(model_dict)
if torch.cuda.is_available():
    model.cuda()
model.eval()

Img_out=[]
for i in range(len(Img)):
    img = Img[i]
    img = img[:,:,0:3]
    img = np.transpose(img,[2,0,1])  #矩阵转置
    if torch.cuda.is_available():
        pic = torch.from_numpy(img).float().cuda().unsqueeze(0)
    else:
        pic = torch.from_numpy(img).float().unsqueeze(0)
    out = model(pic)[-1]
    out = torch.relu(out)[0,0,:,:].cpu().data.numpy()
    out = np.uint8(255 - 255 * out)
    Img_out.append(out)

Im.mimsave('result_Titanic1.gif', Img_out, 'GIF', duration = 1/15)