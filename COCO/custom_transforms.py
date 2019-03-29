import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from PIL import ImageOps

class Denorm(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        return tensor.mul(self.std).add(self.mean)

class ResizeImgAndKeypoints(object):
    def __init__(self, size=224):
        self.size = size
        self.Resize = transforms.Resize((46,46))
    
    def __call__(self, sample):
        im = sample['image']
        keypoints = sample['keypoints'].copy() #2x17x3
        IM_H, IM_W = im.height, im.width
        if(IM_H > IM_W):
            w = int(self.size*IM_W/IM_H)
            h = self.size
            pad_val = int((self.size-w)/2)
            pad = (self.size-w-pad_val,0,pad_val ,0)
            keypoints[:,:,0] = keypoints[:,:,0]*(w/IM_W)
            keypoints[:,:,0][keypoints[:,:,2]>0] += self.size-w-pad_val
            keypoints[:,:,1] = keypoints[:,:,1]*(self.size/IM_H)
        
        else:
            h = int(self.size*IM_H/IM_W)
            w = self.size
            pad_val = int((self.size-h)/2)
            pad = (0,self.size-h-pad_val,0,pad_val)
            keypoints[:,:,0] = keypoints[:,:,0]*(self.size/IM_W)
            keypoints[:,:,1] = keypoints[:,:,1]*(h/IM_H)
            keypoints[:,:,1][keypoints[:,:,2]>0] += self.size-h-pad_val
        
        resized_img = ImageOps.expand(im.resize((w,h),resample=Image.BILINEAR), pad)
        return { 'image' : resized_img , 'image_46x46': self.Resize(resized_img),'keypoints' : keypoints }

class RandomFlipImgAndKeypoints(object):
    def __call__(self, sample):
        img = sample['image']
        keypoints = sample['keypoints']
        
        if np.random.random() > 0.6:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            w, h = img.size
            keypoints[:, :, 0][keypoints[:, :, 2]>0] = w - keypoints[:, :, 0][keypoints[:, :, 2]>0]
            copy = keypoints.copy()
            keypoints[:,1,:], keypoints[:,2,:] = copy[:,2,:], copy[:,1,:]
            keypoints[:,3,:], keypoints[:,4,:] = copy[:,4,:], copy[:,3,:]
            keypoints[:,5,:], keypoints[:,6,:] = copy[:,6,:], copy[:,5,:]
            keypoints[:,7,:], keypoints[:,8,:] = copy[:,8,:], copy[:,7,:]
            keypoints[:,9,:], keypoints[:,10,:] = copy[:,10,:], copy[:,9,:]
            keypoints[:,11,:], keypoints[:,12,:] = copy[:,12,:], copy[:,11,:]
            keypoints[:,13,:], keypoints[:,14,:] = copy[:,14,:], copy[:,13,:]
            keypoints[:,15,:], keypoints[:,16,:] = copy[:,16,:], copy[:,15,:]
        
            return { 'image' : img, 'image_46x46': ImageOps.mirror(sample['image_46x46']) ,'keypoints' : keypoints }
        else: return sample

class ColorJitter(object):
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
        self.tfm = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    
    def __call__(self, sample):
        img = self.tfm(sample['image'])
        return { 'image' : img, 'image_46x46': sample['image_46x46'], 'keypoints': sample['keypoints'] }

class RandomGrayscale(object):
    def __init__(self, p=0.33):
        self.tfm = transforms.RandomGrayscale(p=p)
        self.gs = transforms.Grayscale(num_output_channels=3)
    
    def __call__(self, sample):
        img = self.tfm(sample['image'])
        if(len(img.getbands())<3):
            img = self.gs(img)
            sample['image_46x46'] = self.gs(sample['image_46x46'])	
        return { 'image' : img, 'image_46x46' : sample['image_46x46'], 'keypoints': sample['keypoints'] }

class RandomRotateImgAndKeypoints(object):
    def __init__(self, deg=5):
        self.deg = deg
    
    def __rotate__(self, origin, keypoints, deg):
        ox, oy = origin
        theta = np.math.radians(-deg) #-deg since we measure y,x from top left and not w/2,h/2
        keypoints[:,:,0][keypoints[:,:,2]>0] = (np.math.cos(theta)*(keypoints[:,:,0][keypoints[:,:,2]>0] - ox) - np.math.sin(theta)*(keypoints[:,:,1][keypoints[:,:,2]>0] - oy)) +ox
        keypoints[:,:,1][keypoints[:,:,2]>0] = (np.math.sin(theta)*(keypoints[:,:,0][keypoints[:,:,2]>0] - ox) + np.math.cos(theta)*(keypoints[:,:,1][keypoints[:,:,2]>0] - oy)) +oy
        return keypoints
    
    def __call__(self, sample):
        if(np.random.random()>0.6):
            img = sample['image']
            keypoints = sample['keypoints'].copy()
            rand_deg = np.random.randint(-1*self.deg, self.deg+1)
            img = img.rotate(rand_deg)
            w, h = img.size
            res = self.__rotate__((w/2, h/2), keypoints, rand_deg)
            return { 'image' : img, 'image_46x46' : sample['image_46x46'].rotate(rand_deg) ,'keypoints' : res }
        else:
            return sample

class ToTensor(object):
    def __init__(self):
        self.ToTensor = transforms.ToTensor()
    
    def __call__(self, sample):
        return { 'image' : self.ToTensor(sample['image']),
                 'image_46x46' : self.ToTensor(sample['image_46x46']),
                 'pafs' : torch.tensor(sample['pafs'], dtype=torch.float),
                 'PAF_BINARY_IND' : torch.tensor(sample['PAF_BINARY_IND'], dtype=torch.uint8),
                 'heatmaps' : torch.tensor(sample['heatmaps'], dtype=torch.float),
                 'HM_BINARY_IND' : torch.tensor(sample['HM_BINARY_IND'], dtype=torch.uint8),
                }

class NormalizeImg(object):
    def __init__(self, mean, std):
        self.normalize = transforms.Normalize(mean, std)
    
    def __call__(self, sample):
        sample['image'] = self.normalize(sample['image'])
        return sample

class UnNormalizeImgBatch(object):
    def __init__(self, mean, std):
        self.mean = mean.reshape((1,3,1,1))
        self.std = std.reshape((1,3,1,1))
    
    def __call__(self, batch):
        return (batch*self.std) + self.mean
