import cv2
from tkinter import *
import numpy as np
import time,os
from torch.nn import functional as F
from models import RCF
import torch
import win32ui
from PIL import Image,ImageTk#导入PIL模块中的Image、ImageTk

dlg = win32ui.CreateFileDialog(1)  # 1表示打开文件对话框
dlg.SetOFNInitialDir('C:/')  # 设置打开文件对话框中的初始显示目录
dlg.DoModal()
filepath = dlg.GetPathName()  # 获取选择的文件名称
image_0=Image.open(filepath)#创建Label组件，通过Image=photo设置要展示的图片

img_width= np.size(image_0,1)
img_height= np.size(image_0,0)
scale = min([500/img_width,350/img_height])
img_width_1 = int(img_width*scale)
img_height_1 = int(img_height*scale)
image_00 = image_0.resize((img_width_1, img_height_1), Image.ANTIALIAS)
img_width= np.size(image_00,1)
img_height= np.size(image_00,0)


def show_original_img():
    image_1 = image_0.resize((img_width_1, img_height_1), Image.ANTIALIAS)
    first_image=ImageTk.PhotoImage(image_1)#创建tkinter兼容的图片
    label1.config(text="原始图像：")
    label2.config(image=first_image)
    label2.image=first_image

def show_new_img(label,result):
    result_1 = Image.fromarray(result)
    result = result_1.resize((img_width_1, img_height_1), Image.ANTIALIAS)
    new_img = ImageTk.PhotoImage(result)
    # new_img = ImageTk.PhotoImage(Image.fromarray(result))
    label.config(image=new_img)
    label.image=new_img

def edge_detection():
    label3.config(text="边缘检测图像：")
    img_PIL = Image.open(filepath)#读取数据
    img = np.array(img_PIL)
    
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
    img = np.transpose(img,[2,0,1])  #矩阵转置
    if torch.cuda.is_available():
        pic = torch.from_numpy(img).float().cuda().unsqueeze(0)
    else:
        pic = torch.from_numpy(img).float().unsqueeze(0)
    out = model(pic)[-1]
    out = torch.relu(out)[0,0,:,:].cpu().data.numpy()
    out = np.uint8(255 - 255 * out)
    cv2.imwrite('result_'+os.path.basename(filepath) ,out)
    
    show_new_img(label=label4,result=out)


root = Tk()
root.title("图像边缘检测")  
frame1 = Frame(root,height = img_height+50,width = img_width+50)
frame1.pack(side=LEFT,padx=10,pady=10)
frame2 = Frame(root,height = img_height+50,width = img_width+50)
frame2.pack(side=RIGHT,padx=10,pady=10)
frame3 = Frame(frame1,height = img_height,width = img_width)
frame3.pack()
frame4 = Frame(frame1)
frame4.pack(padx=15,pady=10)
frame5 = Frame(frame2,height = img_height,width = img_width)
frame5.pack()
frame6 = Frame(frame2)
frame6.pack(padx=15,pady=10)

frame1.pack_propagate(0)
frame2.pack_propagate(0)
frame3.pack_propagate(0)
frame5.pack_propagate(0)

label1 = Label(frame3)
label2 = Label(frame3)
label3 = Label(frame5)
label4 = Label(frame5)

label1.pack(padx=10,pady=15)
label2.pack(padx=10,pady=5)
label3.pack(padx=10,pady=15)
label4.pack(padx=10,pady=5)

Button(frame4,text="读取原始图片",command=show_original_img,width=12).pack(anchor=W,padx=5,pady=5)
Button(frame6,text="边缘检测",command=edge_detection,width=12).pack(anchor=W,padx=5,pady=5)
mainloop()
