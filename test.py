# import matplotlib
# matplotlib.use('Agg')
# import os, sys
# import yaml
# from argparse import ArgumentParser
# from tqdm import tqdm

# import imageio
# import numpy as np
# from skimage.transform import resize
# from skimage import img_as_ubyte
# import torch
# import torch.nn.functional as F
# # from sync_batchnorm import DataParallelWithCallback

# from replicate import DataParallelWithCallback

# from generator import OcclusionAwareGenerator
# from modules.keypoint_detector import KPDetector, HEEstimator
# from animate import normalize_kp
# from scipy.spatial import ConvexHull
# from skimage import color


import matplotlib
matplotlib.use('Agg')
import os, sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import cv2
import imageio
# import imageio
import numpy as np
# from skimage.transform import resize
# from skimage import img_as_ubyte
# import torch
# import torch.nn.functional as F
# from sync_batchnorm import DataParallelWithCallback

# from modules.generator import OcclusionAwareGenerator
# from modules.generator import OcclusionAwareGenerator, OcclusionAwareSPADEGenerator

# from modules.keypoint_detector import KPDetector, HEEstimator
# from animate import normalize_kp
# from scipy.spatial import ConvexHull
# from skimage import color

from PIL import Image


# dataloader
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import PIL
from easydict import EasyDict as edict
from torch.autograd import Variable
from torchvision.utils import save_image
import net
import torch




def ToTensor(pic):
    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float()
    else:
        return img

def get_loader(data_config, config, mode="train"):
    # return the DataLoader
    dataset_name = data_config.name
    transform = transforms.Compose([
    transforms.Resize(config.img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    transform_mask = transforms.Compose([
        transforms.Resize(config.img_size, interpolation=PIL.Image.NEAREST),
        ToTensor])
    print(config.data_path)


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    if not requires_grad:
        return Variable(x, requires_grad=requires_grad)
    else:
        return Variable(x)        

def de_norm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)        



# path = '/home/jinliang/BeautyGAN-PyTorch-reimplementation_changed/test_data/non-makeup/'
path = '/home/jinliang/MT_data/images/non-makeup/'


file_path = os.listdir(path)

makeup_path = '/home/jinliang/BeautyGAN-PyTorch-reimplementation_changed/test_data/makeup/style10.jpg'
makeup_image = Image.open(makeup_path)
resize = transforms.Resize([256,256])
ToTensor = transforms.ToTensor()
Normalize = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])




makeup_image = resize(makeup_image)
makeup_image = ToTensor(makeup_image)
makeup_image = Normalize(makeup_image)




for image in file_path:
    print(image)
    image_name = image
    image = Image.open(path + image)
    print(image)
    # image = torch.from_numpy(image)

    image = resize(image)
    image = ToTensor(image)
    image = Normalize(image)

    G = net.Generator_branch(64, 6)
    G.cuda()
    G.load_state_dict(torch.load('/home/jinliang/ljl_makeup_ECCV/snapshot/test15/3_50000_G.pth'))
    G.eval()

    real_org = to_var(image)
    real_ref = to_var(makeup_image)

    real_org = real_org.unsqueeze(dim = 0)
    real_ref = real_ref.unsqueeze(dim = 0)
      
    image_list = []
    # image_list.append(real_org)
    # image_list.append(real_ref)      
    fake_A, fake_B = G(real_org, real_ref)
    
    rec_B, rec_A = G(fake_B, fake_A)
    image_list.append(fake_A)
    # image_list.append(fake_B)

    image_list = torch.cat(image_list, dim=3)
    save_path = os.path.join('./result/{}'.format(image_name))
    save_image(de_norm(image_list.data), save_path, nrow=1, padding=0, normalize=True)












    









