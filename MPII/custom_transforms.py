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

class CenterCrop(object):
    def __init__(self, size=368, p=1):
        self.sz = size
        
    def __call__(self, sample):
        image = sample['image']
        keypoints = sample['keypoints'].copy()
        x,y = sample["meta"]['objpos']
        h, w = image.height, image.width
        crop_h, crop_w = 368, 368
        
        if(w>h):
            y = h/2
            crop_w, crop_h = int(h), int(h)
        else:
            x = w/2
            crop_w, crop_h = int(w), int(w)

        x_l, x_r = x - crop_w//2, x + crop_w//2
        y_t, y_d = y - crop_h//2, y + crop_h//2
        if(x_l<0):
            crop_w -= np.abs(x_l)
            x_l = 0
        elif(x_r>w):
            crop_w -= x_r-w
            x_r = w
        if(y_t<0):
            crop_h -= np.abs(y_t) 
            y_t = 0
        elif(y_d>h):
            crop_h -= y_d-h
            y_d = h
            
        croped_image = image.crop((x_l, y_t, x_r, y_d))  
        keypoints[:,:,:2] = keypoints[:,:,:2] - np.array([[x_l, y_t]])
        keypoints[np.logical_or(keypoints[:,:,0]<0, keypoints[:,:,1]<0)] = np.array([0,0,0])
        
        keypoints[np.logical_or(keypoints[:,:,0]>crop_w, keypoints[:,:,1]>crop_h)] = np.array([0,0,0])

        return { 'image' : croped_image, 'keypoints':keypoints, 'meta': sample["meta"] }
    
class ResizeImgAndKeypoints(object):
    def __init__(self, size=368):
        self.size = size
    
    def __call__(self, sample):
        image = sample['image']
        keypoints = sample['keypoints'].copy().astype(float) #2x17x3
        IM_H, IM_W = image.height, image.width
        cond = np.invert(np.logical_and(keypoints[:,:,0]==0, keypoints[:,:,1]==0))
        if(IM_H > IM_W):
            w = int(self.size*IM_W/IM_H)
            h = self.size
            pad_val = int((self.size-w)/2)
            pad = (self.size-w-pad_val,0,pad_val ,0)
            keypoints[:,:,0] = keypoints[:,:,0]*(w/IM_W)
            keypoints[:,:,0][cond] += self.size-w-pad_val
            keypoints[:,:,1] = keypoints[:,:,1]*(self.size/IM_H)
        
        else:
            h = int(self.size*IM_H/IM_W)
            w = self.size
            pad_val = int((self.size-h)/2)
            pad = (0,self.size-h-pad_val,0,pad_val)
            keypoints[:,:,0] = keypoints[:,:,0]*(self.size/IM_W)
            keypoints[:,:,1] = keypoints[:,:,1]*(h/IM_H)
            keypoints[:,:,1][cond] += self.size-h-pad_val
        
        resized_img = ImageOps.expand(image.resize((w,h),resample=Image.BILINEAR), pad)
        return { 'image' : resized_img, 'keypoints' : keypoints }
    

class FlipHR(object):
    def __init__(self, p=0.25):
        self.p = p
    
    def __call__(self, sample):
        image = sample['image']
        keypoints = sample['keypoints']
        
        if np.random.random() > (1-self.p):
            cond = np.invert(np.logical_and(keypoints[:,:,0]==0, keypoints[:,:,1]==0))
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            w, h = image.size
            keypoints[:, :, 0][cond] = w - keypoints[:, :, 0][cond]
            copy = keypoints.copy()
            keypoints[:,0,:], keypoints[:,5,:] = copy[:,5,:], copy[:,0,:]
            keypoints[:,1,:], keypoints[:,4,:] = copy[:,4,:], copy[:,1,:]
            keypoints[:,2,:], keypoints[:,3,:] = copy[:,3,:], copy[:,2,:]
            keypoints[:,6,:], keypoints[:,11,:] = copy[:,11,:], copy[:,6,:]
            keypoints[:,7,:], keypoints[:,10,:] = copy[:,10,:], copy[:,7,:]
            keypoints[:,8,:], keypoints[:,9,:] = copy[:,9,:], copy[:,8,:]
        
            return { 'image' : image,'keypoints' : keypoints }
        else: return sample

class ColorJitter(object):
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
        self.tfm = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    
    def __call__(self, sample):
        image = self.tfm(sample['image'])
        return { 'image' : image, 'keypoints': sample['keypoints'] }

class RandomGrayscale(object):
    def __init__(self, p=0.33):
        self.tfm = transforms.RandomGrayscale(p=p)
        self.gs = transforms.Grayscale(num_output_channels=3)
    
    def __call__(self, sample):
        image = self.tfm(sample['image'])
        if(len(image.getbands())<3):
            image = self.gs(image)
        return { 'image' : image, 'keypoints': sample['keypoints'] }

class RandomRotateImgAndKeypoints(object):
    def __init__(self, deg=15, p=0.9):
        self.deg = deg
        self.p = p
    
    def __rotate__(self, origin, keypoints, deg, sz, cond):
        ox, oy = origin
        theta = np.math.radians(-deg) #-deg since we measure y,x from top left and not w/2,h/2
        X = keypoints[:,:,0][cond]
        Y = keypoints[:,:,1][cond]
        
        keypoints[:,:,0][cond] = ox + (np.math.cos(theta)*(X - ox) - np.math.sin(theta)*(Y - oy)) 
        keypoints[:,:,1][cond] = oy + (np.math.sin(theta)*(X - ox) + np.math.cos(theta)*(Y - oy)) 
        
        inds = np.logical_or(np.any((keypoints[:,:,:2]<0), axis=2), np.any((keypoints[:,:,:2]>sz), axis=2))
        keypoints[inds,:] = np.array([0,0,0])
        return keypoints
    
    def __call__(self, sample):
        if(np.random.random()>(1-self.p)):
            image = sample['image']
            keypoints = sample['keypoints'].copy()
            cond = np.invert(np.logical_and(keypoints[:,:,0]==0, keypoints[:,:,1]==0))
            rand_deg = np.random.randint(-1*self.deg, self.deg+1)
            image = image.rotate(rand_deg)
            w, h = image.size
            res = self.__rotate__((w/2, h/2), keypoints, rand_deg, h, cond)
            return { 'image' : image, 'keypoints' : res }
        else:
            return sample

class ToTensor(object):
    def __init__(self):
        self.ToTensor = transforms.ToTensor()
    
    def __call__(self, sample):
        return { 'image' : self.ToTensor(sample['image']),
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

class Resize(object):
    def __init__(self, size=368):
        self.size = size
    
    def __call__(self, im):
        if(im.height > im.width):
            w = int(self.size*im.width/im.height)
            h = self.size
            pad_val = int((self.size-w)/2)
            pad = (self.size-w-pad_val,0,pad_val,0)
        else:
            h = int(self.size*im.height/im.width)
            w = self.size
            pad_val = int((self.size-h)/2)
            pad = (0,self.size-h-pad_val,0,pad_val)
        return ImageOps.expand(im.resize((w,h),resample=Image.BILINEAR), pad)
