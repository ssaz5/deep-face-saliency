from skimage.data import data_dir
from skimage.util import img_as_ubyte
from skimage.morphology import opening, disk
from skimage import io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import scipy as sp
import scipy.stats as st
import os
from PIL import Image
import cv2

def gkern(kernlen_x=21,kernlen_y=21, nsig_x=3, nsig_y=4):
    """Returns a 2D Gaussian kernel array."""

    interval_x = (2*nsig_x+1.)/(kernlen_x)
    interval_y = (2*nsig_y+1.)/(kernlen_y)
    x = np.linspace(-nsig_x-interval_x/2., nsig_x+interval_x/2., kernlen_x+1)
    y = np.linspace(-nsig_y-interval_y/2., nsig_y+interval_y/2., kernlen_y+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.diff(st.norm.cdf(y))
    kernel_raw = np.sqrt(np.outer(kern1d, kern2d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel    

def opening_transform(preds, disk_size=10):    
    pred = img_as_ubyte(preds.cpu().numpy().squeeze(0))
    selem = disk(disk_size)
    filtered_pred = opening(pred, selem)
    filtered_pred *= 255
    filtered_pred_pil = Image.fromarray(filtered_pred)
    filtered_pred_pil.save('filtered_pred_pil.png')
    return filtered_pred_pil
    
def pass_img(model, img):
    img = Image.open(img).convert('RGB')
    to_tensor = transforms.ToTensor()
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.5, 0.5, 0.5])
    res = transforms.Resize([240, 320])
    img = res(img)        
    img = to_tensor(img)
    img = Variable(img.unsqueeze(0))
    img.cuda()
    out = model.forward(img.type(torch.cuda.FloatTensor))
    _, preds = torch.max(out, 1)
    return preds, out

def pass_img_fcn(model, img):
    img = Image.open(img).convert('RGB')
    to_tensor = transforms.ToTensor()
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.5, 0.5, 0.5])
    res = transforms.Resize([240, 320])
    img = res(img)        
    img = to_tensor(img)
    img = Variable(img.unsqueeze(0))
    img.cuda()
    out = model.forward(img.type(torch.cuda.FloatTensor))
    out = out.squeeze(0)
    _, preds = torch.max(out, 0)
    return preds, out

def pass_img_vae(model, img):
    img = Image.open(img).convert('RGB')
    to_tensor = transforms.ToTensor()
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.5, 0.5, 0.5])
    res = transforms.Resize([240, 320])
    img = res(img)        
    img = to_tensor(img)
    img = Variable(img.unsqueeze(0))
    img.cuda()
    
    return model.forward(img.type(torch.cuda.FloatTensor))


def get_blobs(img_src):
    img_src = cv2.imread(img_src)
    img_src = cv2.resize(img_src, (320, 240))
    filtered_pred_cv = cv2.imread('filtered_pred_pil.png', cv2.IMREAD_GRAYSCALE)
    filtered_pred_cv = cv2.bitwise_not(filtered_pred_cv)
    filtered_pred_cv[filtered_pred_cv!=255] =0
    blur = cv2.blur(filtered_pred_cv, (15,15), 0)
    params = cv2.SimpleBlobDetector_Params() 
    params.minThreshold = 10
    params.filterByArea = True
#     params.filterByCircularity = False
#     params.filterByInertia = False
#     params.filterByConvexity = False
    params.minArea = 15
    params.maxArea = 1e10
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3 :
        detector = cv2.SimpleBlobDetector(params)
    else: 
        detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(blur)
    im_with_keypoints = cv2.drawKeypoints(img_src, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return keypoints, im_with_keypoints

def get_faces(img, keypoints, pred=None):
    img = Image.open(img).convert('RGB')
    res = transforms.Resize([240, 320])
    topil = transforms.ToPILImage(mode=None)
    to_tensor = transforms.ToTensor()
    img = res(img) 
    faces = []
    for keypoint in keypoints:
        x, y = keypoint.pt
        diam = keypoint.size/2
        box_x1 = x - diam
        box_x2 = x + diam
        box_y1 = y - diam*1.2
        box_y2 = y + diam*1.2
        face = img.crop((box_x1, box_y1, box_x2, box_y2))
        if pred:
            mask = pred.crop((box_x1, box_y1, box_x2, box_y2))
            mask = to_tensor(mask).float()
        else:
            mask = gkern(face.size[1], face.size[0], 2, 2.5)
            mask = mask / np.max(mask)
            th = (np.mean(mask) - 1*np.std(mask)) 
            mask[mask<th] = 0
            mask[mask>=th] = 1
            mask = np.tile(mask,[3,1,1])
            mask = torch.from_numpy(mask).float()
        face = to_tensor(face).float()
        face = torch.mul(face, mask)
        face = topil(face)
        faces.append(face)
    return faces
        