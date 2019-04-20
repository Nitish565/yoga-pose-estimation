import numpy as np
from PIL import Image
from CONSTANTS import *
import torch
import torch.nn.functional as F
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import scipy.ndimage.filters as fi
from skimage.feature import peak_local_max
import time


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts)*1000))
        return result
    return timed

def calculate_heatmap_optimized(fliped_img, kp_id, keypoints):
    pad = 7
    ps = 15
    g_vals = GAUSSIAN_15X15
    if(kp_id in SMALLER_HEATMAP_GROUP):
        ps = 9
        g_vals = GAUSSIAN_9X9
    
    ps_hf = ps//2
    
    points = keypoints[:,kp_id, :2][keypoints[:,kp_id,2]>0]
    points = np.rint(points).astype(int)
    KEYPOINT_EXISTS = (len(points)>0)
    ncols, nrows = fliped_img.shape[:2]
    mask = np.zeros((ncols, nrows))
    
    for (x,y) in points:
        mask[x-ps_hf : x+ps_hf+1, y-ps_hf : y+ps_hf+1] = g_vals

    mask = mask[pad:-pad, pad:-pad]
    return mask, KEYPOINT_EXISTS

def get_heatmap_masks_optimized(img, keypoints, kp_ids = KEYPOINT_ORDER):
    img = np.array(img)
    h,w = img.shape[:2]
    pad = 7
    img = np.pad(img, pad_width=[(pad,pad),(pad,pad),(0,0)], mode='constant', constant_values=0)
    
    heatmaps = np.zeros((len(kp_ids)+1, h, w))
    HM_BINARY_IND = np.zeros(len(kp_ids)+1)
    fliped_img = img.transpose((1,0,2))
    kps_copy = keypoints.copy()
    kps_copy[:,:,:2][kps_copy[:,:,2]>0] = kps_copy[:,:,:2][kps_copy[:,:,2]>0]+pad
    
    for i, kp_id in enumerate(kp_ids):
        mask, HM_IS_LABELED = calculate_heatmap_optimized(fliped_img, kp_id, kps_copy)
        HM_BINARY_IND[i] = int(HM_IS_LABELED)
        mask = mask.transpose()
        heatmaps[i] = mask
    heatmaps[len(kp_ids)] = np.ones((h,w)) - np.sum(heatmaps, axis=0)
    HM_BINARY_IND[len(kp_ids)] = 1
    return heatmaps, HM_BINARY_IND

def calculate_paf_mask_optimized(fliped_img, joint_pair, keypoints, limb_width):
    #(img) HxWx3 to (fliped_img) WxHx3 (x,y,3)
    j1_idx, j2_idx = joint_pair[0], joint_pair[1]
    
    ncols_x, nrows_y  = 46,46#fliped_img.shape[:2]
    mask = np.zeros((ncols_x, nrows_y))               #in x,y order
    col, row = np.ogrid[:ncols_x, :nrows_y]
    
    paf_p_x = np.zeros((len(keypoints), ncols_x, nrows_y))
    paf_p_y = np.zeros((len(keypoints), ncols_x, nrows_y))
    NON_ZERO_VEC_COUNT = np.zeros((2, ncols_x, nrows_y))
    PAF_IND = False
    final_paf_map = np.zeros((2, ncols_x, nrows_y))
    
    for i, item in enumerate(keypoints):
        j1, j2 =  item[j1_idx][:2], item[j2_idx][:2]
        keypoints_detected = item[j1_idx][2] and item[j2_idx][2]
        PAF_IND = PAF_IND or keypoints_detected>0
        
        if(keypoints_detected):
            limb_length = np.linalg.norm(j2 - j1)
            if(limb_length>1e-8):
                v = (j2 - j1)/limb_length
                v_perp = np.array([v[1], -v[0]])
                center_point = (j1 + j2)/2
                
                cond1 = np.abs(np.dot(v, np.array([col, row]) - center_point))<= limb_length/2
                cond2 = np.abs(np.dot(v_perp, np.array([col, row]) - j1))<=limb_width
                mask = np.logical_and(cond1, cond2)
                paf_p_x[i], paf_p_y[i] = mask*v[0], mask*v[1]
                if(v[0]):
                    NON_ZERO_VEC_COUNT[0][mask] +=1
                if(v[1]):
                    NON_ZERO_VEC_COUNT[1][mask] +=1

    NON_ZERO_VEC_COUNT[NON_ZERO_VEC_COUNT==0] = 1
    final_paf_map[0], final_paf_map[1] = (paf_p_x.sum(axis=0)/NON_ZERO_VEC_COUNT[0]), (paf_p_y.sum(axis=0)/NON_ZERO_VEC_COUNT[1])
    return final_paf_map, PAF_IND


def get_paf_masks_optimized(img, keypoints, part_pairs=SKELETON, limb_width=5):
    img = np.array(img)
    h,w = 46,46
    pafs = np.zeros((len(part_pairs)*2, h, w))
    PAF_BINARY_IND = np.zeros(len(part_pairs)*2)
    fliped_img = img.transpose((1,0,2))
    kps_copy = keypoints.copy()
    kps_copy[:,:,:2][kps_copy[:,:,2]>0] = kps_copy[:,:,:2][kps_copy[:,:,2]>0]*0.125
    
    for i, joint_pair in enumerate(part_pairs):
        mask, PAF_IS_LABELED = calculate_paf_mask_optimized(fliped_img, joint_pair, kps_copy, 0.125*limb_width)
        PAF_BINARY_IND[2*i], PAF_BINARY_IND[(2*i)+1]  = int(PAF_IS_LABELED), int(PAF_IS_LABELED)
        mask = mask.transpose((0,2,1))
        pafs[2*i], pafs[(2*i) +1] = mask[0], mask[1]   #x component, y component of v
    return pafs, PAF_BINARY_IND

def get_keypoints_from_annotations(anns):
    keypoints = []
    for ann in anns:
        keypoints.append(list(zip(ann['keypoints'][::3], ann['keypoints'][1::3], ann['keypoints'][2::3])))
    keypoints = np.array(keypoints).astype(float)
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

def print_training_loss_summary(loss, total_steps, current_epoch, n_epochs, n_batches, print_every=10):
    #prints loss at the start of the epoch, then every 10(print_every) steps taken by the optimizer
    steps_this_epoch = (total_steps%n_batches)
    
    if(steps_this_epoch==1 or steps_this_epoch%print_every==0):
        print ('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}'
               .format(current_epoch, n_epochs, steps_this_epoch, n_batches, loss))

def paf_and_heatmap_loss(pred_pafs_stages, pafs_gt, paf_inds, pred_hms_stages, hms_gt, hm_inds):
    cumulative_paf_loss = 0
    cumulative_hm_loss = 0
    '''
    for paf_stg in pred_pafs_stages:
            #scaled_pafs = F.interpolate(paf_stg, 368, mode="bilinear", align_corners=True).to(device)
        stg_paf_loss = torch.dist(paf_stg[paf_inds], pafs_gt[paf_inds])
        cumulative_paf_loss += stg_paf_loss
    '''
    for hm_stg in pred_hms_stages:
        scaled_hms = F.interpolate(hm_stg, 368, mode="bilinear", align_corners=True).to(device)
        stg_hm_loss = torch.dist(scaled_hms[hm_inds], hms_gt[hm_inds])
        cumulative_hm_loss += stg_hm_loss
   
    return cumulative_paf_loss+cumulative_hm_loss

def get_peaks(part_heatmap, nms_window=10):
    coords = peak_local_max(part_heatmap, min_distance=nms_window)
    coords[:,[0, 1]] = coords[:,[1, 0]]
    return coords

def gkern2(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    
    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen//2, kernlen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return fi.gaussian_filter(inp, nsig)


'''
def calculate_heatmap(fliped_img, kp_id, keypoints, sigma=7):
    if(kp_id in SMALLER_HEATMAP_GROUP):
        sigma = 0.5*sigma
    
    points = keypoints[:,kp_id, :2][keypoints[:,kp_id,2]>0]
    KEYPOINT_EXISTS = (len(points)>0)
    ncols, nrows = fliped_img.shape[:2]
    col, row = np.ogrid[:ncols, :nrows]
    mask = np.zeros((ncols, nrows))
    
    for point in points:
        mask = np.maximum(mask, np.exp(-np.linalg.norm(np.array([col, row]) - point)**2 / sigma**2))
    mask[mask<1e-4] = 0
    return mask, KEYPOINT_EXISTS #w,h (x,y)

#@timeit
def get_heatmap_masks(img, keypoints, kp_ids = KEYPOINT_ORDER, sigma=7):
    img = np.array(img)
    h,w = img.shape[:2]
    heatmaps = np.zeros((len(kp_ids)+1, h, w))
    HM_BINARY_IND = np.zeros(len(kp_ids)+1)
    fliped_img = img.transpose((1,0,2))
    
    for i, kp_id in enumerate(kp_ids):
        mask, HM_IS_LABELED = calculate_heatmap(fliped_img, kp_id, keypoints, sigma)
        HM_BINARY_IND[i] = int(HM_IS_LABELED)
        mask = mask.transpose()
        heatmaps[i] = mask
    heatmaps[len(kp_ids)] = np.ones((h,w)) - np.sum(heatmaps, axis=0)
    HM_BINARY_IND[len(kp_ids)] = 1
    return heatmaps, HM_BINARY_IND
'''

'''
def calculate_paf_mask(fliped_img, joint_pair, keypoints, limb_width=5):
    #(img) HxWx3 to (fliped_img) WxHx3 (x,y,3)
    j1_idx, j2_idx = joint_pair[0], joint_pair[1]
    
    ncols_x, nrows_y  = fliped_img.shape[:2]
    mask = np.zeros((ncols_x, nrows_y))               #in x,y order
    col, row = np.ogrid[:ncols_x, :nrows_y]
    
    paf_p_x = np.zeros((len(keypoints), ncols_x, nrows_y))
    paf_p_y = np.zeros((len(keypoints), ncols_x, nrows_y))
    NON_ZERO_VEC_COUNT = np.zeros((2, ncols_x, nrows_y))
    PAF_IND = False
    final_paf_map = np.zeros((2, ncols_x, nrows_y))
    
    for i, item in enumerate(keypoints):
        j1, j2 =  item[j1_idx][:2], item[j2_idx][:2]
        keypoints_detected = item[j1_idx][2] and item[j2_idx][2]
        PAF_IND = PAF_IND or keypoints_detected>0
        
        if(keypoints_detected):
            limb_length = np.linalg.norm(j2 - j1)
            if(limb_length>1e-2):
                v = (j2 - j1)/limb_length
                v_perp = np.array([v[1], -v[0]])
                center_point = (j1 + j2)/2
                
                cond1 = np.abs(np.dot(v, np.array([col, row]) - center_point))<= limb_length/2
                cond2 = np.abs(np.dot(v_perp, np.array([col, row]) - j1))<=limb_width
                mask = np.logical_and(cond1, cond2)
                paf_p_x[i], paf_p_y[i] = mask*v[0], mask*v[1]
                if(v[0]):
                    NON_ZERO_VEC_COUNT[0][mask] +=1
                if(v[1]):
                    NON_ZERO_VEC_COUNT[1][mask] +=1

    NON_ZERO_VEC_COUNT[NON_ZERO_VEC_COUNT==0] = 1
    final_paf_map[0], final_paf_map[1] = (paf_p_x.sum(axis=0)/NON_ZERO_VEC_COUNT[0]), (paf_p_y.sum(axis=0)/NON_ZERO_VEC_COUNT[1])
    return final_paf_map, PAF_IND

#@timeit
def get_paf_masks(img, keypoints, part_pairs=SKELETON, limb_width=5):
    img = np.array(img)
    h,w = img.shape[:2]
    pafs = np.zeros((len(part_pairs)*2, h, w))
    PAF_BINARY_IND = np.zeros(len(part_pairs)*2)
    fliped_img = img.transpose((1,0,2))
    
    for i, joint_pair in enumerate(part_pairs):
        mask, PAF_IS_LABELED = calculate_paf_mask(fliped_img, joint_pair, keypoints, limb_width)
        PAF_BINARY_IND[2*i], PAF_BINARY_IND[(2*i)+1]  = int(PAF_IS_LABELED), int(PAF_IS_LABELED)
        mask = mask.transpose((0,2,1))
        pafs[2*i], pafs[(2*i) +1] = mask[0], mask[1]   #x component, y component of v
    return pafs, PAF_BINARY_IND
'''


'''
def calculate_heatmap_46x46(fliped_img, kp_id, keypoints):
    points = keypoints[:,kp_id, :2][keypoints[:,kp_id,2]>0]
    KEYPOINT_EXISTS = (len(points)>0)
    ncols, nrows = fliped_img.shape[:2]
    mask = np.zeros((ncols, nrows))
    
    pad = 1
    ps = 1
    g_vals = HM_PATCH_1x1
    for (x,y) in points:
        mask[x, y] = g_vals

    mask = mask[pad:-pad, pad:-pad]
    return mask, KEYPOINT_EXISTS

def get_heatmap_masks_46x46(img, keypoints, kp_ids = KEYPOINT_ORDER):
    img = np.array(img)
    h,w = 46,46#img.shape[:2]
    pad = 1
    img = np.pad(img, pad_width=[(pad,pad),(pad,pad),(0,0)], mode='constant', constant_values=0)
    
    heatmaps = np.zeros((len(kp_ids)+1, h, w))
    HM_BINARY_IND = np.zeros(len(kp_ids)+1)
    fliped_img = img.transpose((1,0,2))
    kps_copy = keypoints.copy()
    kps_copy[:,:,:2][kps_copy[:,:,2]>0] = (kps_copy[:,:,:2][kps_copy[:,:,2]>0]*0.125)+pad
    kps_copy = np.rint(kps_copy).astype(int)
    print(kps_copy)
    for i, kp_id in enumerate(kp_ids):
        mask, HM_IS_LABELED = calculate_heatmap_46x46(fliped_img, kp_id, kps_copy)
        HM_BINARY_IND[i] = int(HM_IS_LABELED)
        mask = mask.transpose()
        heatmaps[i] = mask
    heatmaps[len(kp_ids)] = np.ones((h,w)) - np.sum(heatmaps, axis=0)
    HM_BINARY_IND[len(kp_ids)] = 1
    return heatmaps, HM_BINARY_IND
'''