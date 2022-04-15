from torch.nn import functional as F
from models import RCF
import cv2
import torch
import numpy as np

img_path = 'hwf_1.jpg'
img = cv2.imread(img_path)
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
# mean_bgr = np.array([104.00699, 116.66877, 122.67892])
# img = img - mean_bgr
img = np.transpose(img,[2,0,1])  #矩阵转置
if torch.cuda.is_available():
    pic = torch.from_numpy(img).float().cuda().unsqueeze(0)
else:
    pic = torch.from_numpy(img).float().unsqueeze(0)
out = model(pic)[-1]
out = torch.relu(out)[0,0,:,:].cpu().data.numpy()
#out = F.sigmoid(out)[0,0,:,:].cpu().data.numpy()
#out = out - 0.5
out = np.uint8(255 - 255 * out)
cv2.imwrite('result_hwf.jpg',out)
#cv2.imwrite("D://wangyang//face1" + "/" + filename, img)
#cv2.imencode('.jpg', out)[1].tofile('NYUD/2.jpg')
