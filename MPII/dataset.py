import torch
from PIL import Image
import time
import model_utils
from model_utils import timeit
import numpy as np

class MPII_Person_Dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, annotations, tfms, tensor_tfms, im_sz=368):
        super(MPII_Person_Dataset, self).__init__()
        self.image_dir = image_dir
        self.annotations = np.load(annotations, allow_pickle=True)
        
        self.tfms = tfms
        self.tensor_tfms = tensor_tfms
        self.get_heatmap_masks = model_utils.get_heatmap_masks_optimized     
        self.get_paf_masks = model_utils.get_paf_masks_optimized
        self.limb_width = 5
        self.sigma = 7
        self.len = len(self.annotations)
        self.heatmap_ps_map = model_utils.get_heatmap_ps_map()
            
    #@timeit
    def __getitem__(self, index):
        ann = self.annotations[index]
        image = Image.open('./images/'+ann['img_paths'])
        keypoints = model_utils.get_keypoints_from_annotations(ann)
        meta = {}
        meta["objpos"] = ann["objpos"]
        if self.tfms:
            tfmd_sample = self.tfms({"image":image, "keypoints":keypoints, "meta" : meta})
            image, keypoints = tfmd_sample["image"], tfmd_sample["keypoints"]
        
        heatmaps, HM_BINARY_IND = self.get_heatmap_masks(keypoints, self.sigma, self.heatmap_ps_map)
        pafs, PAF_BINARY_IND = self.get_paf_masks(keypoints, limb_width=self.limb_width) 
            
        if self.tensor_tfms:
            res = self.tensor_tfms({"image":image, "pafs":pafs, "PAF_BINARY_IND":PAF_BINARY_IND, "heatmaps":heatmaps, "HM_BINARY_IND":HM_BINARY_IND})
            image = res["image"]
            pafs = res["pafs"]
            PAF_BINARY_IND = res["PAF_BINARY_IND"]
            heatmaps = res["heatmaps"]
            HM_BINARY_IND = res["HM_BINARY_IND"]
        return (image, pafs, PAF_BINARY_IND, heatmaps, HM_BINARY_IND)
    
    def __len__(self):
        return self.len

