import torch
import skimage
import skimage.io as io
import skimage.transform, skimage.util
from PIL import Image
import model_utils

class COCO_Person_Dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, img_id_to_image_info, img_id_to_annotations, tfms, sigma=4, limb_width=3):
        super(COCO_Person_Dataset, self).__init__()
        self.image_dir = image_dir
        self.img_id_to_image_info = img_id_to_image_info
        self.img_id_to_annotations = img_id_to_annotations
        self.im_ids = list(img_id_to_annotations.keys())
        self.tfms = tfms
        self.get_heatmap_masks = model_utils.get_heatmap_masks
        self.get_paf_masks = model_utils.get_paf_masks
        self.limb_width = limb_width
        self.sigma = sigma
        self.len = len(self.im_ids)
    
    def __getitem__(self, index):
        im_id = self.im_ids[index]
        img = Image.open(self.image_dir+self.img_id_to_image_info[im_id]['file_name'])
        annotations = self.img_id_to_annotations[im_id]
        keypoints = model_utils.get_keypoints_from_annotations(annotations)
        
        if self.tfms:
            tfmd_sample = self.tfms({"image":img, "keypoints":keypoints})
            img, keypoints = tfmd_sample["image"], tfmd_sample["keypoints"]
        
        heatmaps, HM_BINARY_IND = self.get_heatmap_masks(img, keypoints, sigma=self.sigma)
        pafs, PAF_BINARY_IND = self.get_paf_masks(img, keypoints, limb_width=self.limb_width)
        return (img, heatmaps, pafs, HM_BINARY_IND, PAF_BINARY_IND)
    
    def __len__(self):
        return self.len

#img = io.imread(self.image_dir+self.img_id_to_image_info[im_id]['file_name'])
