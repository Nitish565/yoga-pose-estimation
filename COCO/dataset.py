import torch
from PIL import Image
import time
import model_utils
from model_utils import timeit
import numpy as np


class COCO_Person_Dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, processed_files, tfms, tensor_tfms, sigma=7, limb_width=5):
        super(COCO_Person_Dataset, self).__init__()
        self.image_dir = image_dir
        self.im_ids = np.load(processed_files["im_ids"])
        self.img_id_to_annotations = np.load(processed_files["img_id_to_annotations"]).ravel()[0]
        self.img_id_to_image_info = np.load(processed_files["img_id_to_image_info"]).ravel()[0]
        
        self.tfms = tfms
        self.tensor_tfms = tensor_tfms
        self.get_heatmap_masks = model_utils.get_heatmap_masks
        self.get_paf_masks = model_utils.get_paf_masks
        self.limb_width = limb_width
        self.sigma = sigma
        self.len = len(self.im_ids)

    #@timeit
    def __getitem__(self, index):
        im_id = self.im_ids[index]
        img = Image.open(self.image_dir+self.img_id_to_image_info[im_id]['file_name'])
        annotations = self.img_id_to_annotations[im_id]
        keypoints = model_utils.get_keypoints_from_annotations(annotations)
        
        if self.tfms:#~5-8ms with minimal tfms, ~20ms if all included
            tfmd_sample = self.tfms({"image":img, "keypoints":keypoints})
            img, keypoints = tfmd_sample["image"], tfmd_sample["keypoints"]
        
        heatmaps, HM_BINARY_IND = [],[]#self.get_heatmap_masks(img, keypoints, sigma=self.sigma)
        pafs, PAF_BINARY_IND = self.get_paf_masks(img, keypoints, limb_width=self.limb_width)
        
        if self.tensor_tfms:#~23ms
            res = self.tensor_tfms({"image":img, "pafs":pafs, "PAF_BINARY_IND":PAF_BINARY_IND, "heatmaps":heatmaps, "HM_BINARY_IND":HM_BINARY_IND})
            img = res["image"]
            pafs = res["pafs"]
            PAF_BINARY_IND = res["PAF_BINARY_IND"]
            heatmaps = res["heatmaps"]
            HM_BINARY_IND = res["HM_BINARY_IND"]
        return (img, pafs, PAF_BINARY_IND, heatmaps, HM_BINARY_IND)
    
    def __len__(self):
        return self.len

#img = io.imread(self.image_dir+self.img_id_to_image_info[im_id]['file_name'])
