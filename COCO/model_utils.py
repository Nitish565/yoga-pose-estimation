import numpy as np
from PIL import Image
from CONSTANTS import *

def get_joint_positions(joint_type, keypoints, keypoint_type_to_idx):
    res = []
    idx = keypoint_type_to_idx[joint_type]
    for item in keypoints:
        if(item[idx][2]!=0):
            res.append(item[idx][:2])
    return np.array(res)

def calculate_heatmap(img, joint_type, keypoints, keypoint_type_to_idx, sigma=7):
    # HxWx3 to WxHx3 (x,y,3)
    fliped_img = img.transpose((1,0,2))
    points = get_joint_positions(joint_type, keypoints, keypoint_type_to_idx)
    KEYPOINT_EXISTS = (len(points)>0)
    ncols, nrows = fliped_img.shape[:2]
    col, row = np.ogrid[:ncols, :nrows]
    mask = np.zeros((ncols, nrows))
    
    for point in points:
        mask = np.maximum(mask, np.exp(-np.linalg.norm(np.array([col, row]) - point)**2 / sigma**2))

    return mask, KEYPOINT_EXISTS #w,h (x,y)

def calculate_paf_mask(img, joint_pair, keypoints, keypoint_type_to_idx, limb_width=5):
    # HxWx3 to WxHx3 (x,y,3)
    fliped_img = img.transpose((1,0,2))
    j1_idx, j2_idx = keypoint_type_to_idx[joint_pair[0]], keypoint_type_to_idx[joint_pair[1]]
    
    ncols_x, nrows_y  = fliped_img.shape[:2]
    mask = np.zeros((ncols_x, nrows_y))               #in x,y order
    col, row = np.ogrid[:ncols_x, :nrows_y]
    
    for item in keypoints:
        j1, j2 =  item[j1_idx][:2], item[j2_idx][:2]
        keypoints_detected = item[j1_idx][2] and item[j2_idx][2]
        
        if(keypoints_detected):
            limb_length = np.linalg.norm(j2 - j1)
            v = (j2 - j1)/limb_length
            v_perp = np.array([v[1], -v[0]])
            center_point = (j1 + j2)/2
            
            cond1 = np.abs(np.dot(v, np.array([col, row]) - center_point))<= limb_length/2
            cond2 = np.abs(np.dot(v_perp, np.array([col, row]) - j1))<=limb_width
            mask = np.maximum(mask, np.logical_and(cond1, cond2))
        else:
            v = np.array([0,0])
    #return mask
    paf_map = np.zeros((2, ncols_x, nrows_y))
    paf_map[0], paf_map[1] = np.copy(mask)*v[0], np.copy(mask)*v[1]
    return paf_map, keypoints_detected>0


def get_heatmap_masks(img, keypoints, keypoint_labels=keypoint_labels, keypoint_type_to_idx=keypoint_type_to_idx, sigma=7):
    img = np.array(img)
    h,w = img.shape[:2]
    heatmaps = np.zeros((len(keypoint_labels), h, w))
    HM_BINARY_IND = np.zeros(len(keypoint_labels))
    
    for i, joint_type in enumerate(keypoint_labels):
        mask, HM_IS_LABELED = calculate_heatmap(img, joint_type, keypoints, keypoint_type_to_idx, sigma)
        HM_BINARY_IND[i] = int(HM_IS_LABELED)
        mask = mask.transpose()
        heatmaps[i] = mask
    return heatmaps, HM_BINARY_IND

def get_paf_masks(img, keypoints, joint_pairs=part_pairs, keypoint_type_to_idx=keypoint_type_to_idx, limb_width=5):
    img = np.array(img)
    h,w = img.shape[:2]
    pafs = np.zeros((len(part_pairs)*2, h, w))
    PAF_BINARY_IND = np.zeros(len(part_pairs)*2)
    
    for i, joint_pair in enumerate(part_pairs):
        mask, PAF_IS_LABELED = calculate_paf_mask(img, joint_pair, keypoints, keypoint_type_to_idx, limb_width)
        PAF_BINARY_IND[2*i], PAF_BINARY_IND[(2*i)+1]  = int(PAF_IS_LABELED), int(PAF_IS_LABELED)
        mask = mask.transpose((0,2,1))
        pafs[2*i], pafs[(2*i) +1] = mask[0], mask[1]   #x component, y component of v
    return pafs, PAF_BINARY_IND

def get_keypoints_from_annotations(anns):
    keypoints = []
    for ann in anns:
        keypoints.append(list(zip(ann['keypoints'][::3], ann['keypoints'][1::3], ann['keypoints'][2::3])))
    keypoints = np.array(keypoints)
    return keypoints

def freeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = True    

def freeze_other_paf_stages(model, stg):
    if(stg==1):
        unfreeze_all_layers(model.Stage1)
        freeze_all_layers(model.Stage2)
        freeze_all_layers(model.Stage3)
        freeze_all_layers(model.Stage4)
    elif(stg==2):
        freeze_all_layers(model.Stage1)
        unfreeze_all_layers(model.Stage2)
        freeze_all_layers(model.Stage3)
        freeze_all_layers(model.Stage4)
    elif(stg==3):
        freeze_all_layers(model.Stage1)
        freeze_all_layers(model.Stage2)
        unfreeze_all_layers(model.Stage3)
        freeze_all_layers(model.Stage4)
    elif(stg==4):
        freeze_all_layers(model.Stage1)
        freeze_all_layers(model.Stage2)
        freeze_all_layers(model.Stage3)
        unfreeze_all_layers(model.Stage4)

def unfreeze_all_paf_stages(model):
    unfreeze_all_layers(model.Stage1)
    unfreeze_all_layers(model.Stage2)
    unfreeze_all_layers(model.Stage3)
    unfreeze_all_layers(model.Stage4)

def freeze_all_paf_stages(model):
    freeze_all_layers(model.Stage1)
    freeze_all_layers(model.Stage2)
    freeze_all_layers(model.Stage3)
    freeze_all_layers(model.Stage4)

def freeze_other_hm_stages(model, stg):
    if(stg==1):
        unfreeze_all_layers(model.Stage1)
        freeze_all_layers(model.Stage2)
    elif(stg==2):
        freeze_all_layers(model.Stage1)
        unfreeze_all_layers(model.Stage2)

def unfreeze_all_hm_stages(model):
    unfreeze_all_layers(model.Stage1)
    unfreeze_all_layers(model.Stage2)

def freeze_all_hm_stages(model):
    freeze_all_layers(model.Stage1)
    freeze_all_layers(model.Stage2)
