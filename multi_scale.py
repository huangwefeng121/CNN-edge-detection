from torch.nn import functional as F
import cv2
import torch
import numpy as np
import torch.nn as nn
from models_3 import RCF

img_path = '2.jpg'
img_ = cv2.imread(img_path)
model_path = './pretrained/checkpoint_epoch0_3.pth'
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
H_, W_, _ = img_.shape
multi_fuse = np.zeros((H_, W_), np.float32)

scale = [0.5, 1, 1.5]
for k in range(0, len(scale)):
    img = cv2.resize(img_, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
    img = np.transpose(img,[2,0,1])  #矩阵转置
    if torch.cuda.is_available():
        pic = torch.from_numpy(img).float().cuda().unsqueeze(0)
    else:
        pic = torch.from_numpy(img).float().unsqueeze(0)
    out = model(pic)[-1]
    out = torch.relu(out)[0,0,:,:].cpu().data.numpy()
    out = np.uint8(255 - 255 * out)
    add = 'result' + str(k) + '.jpg'
    cv2.imwrite(add,out)
    if scale[k] > 1:
        out = cv2.resize(out, None, fx=1/scale[k], fy=1/scale[k], interpolation=cv2.INTER_LINEAR)
        add = 'result_' + str(k) + '.jpg'
        cv2.imwrite(add,out)
    elif scale[k] < 1:
        out = cv2.resize(out, None, fx=1/scale[k], fy=1/scale[k], interpolation=cv2.INTER_LINEAR)
        add = 'result_' + str(k) + '.jpg'
        cv2.imwrite(add,out)
    multi_fuse += out
multi_fuse = multi_fuse / len(scale)
cv2.imwrite('result_fuse',multi_fuse)