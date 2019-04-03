import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import model_utils
from CONSTANTS import *

def hide_subplot_axes(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

def plot_paf_maps_from_annotations(img, keypoints, joint_pairs=part_pairs, keypoint_type_to_idx=keypoint_type_to_idx, n_items=19, figsize=(16,12), limb_width=5):
    fig, axes = plt.subplots(5, 4, figsize=figsize)
    
    for i,ax in enumerate(axes.flat):
        hide_subplot_axes(ax)
        if(i<19):
            ax.text(10,10, joint_pairs[i][0]+'->'+joint_pairs[i][1], va='top', color="white", fontsize=12)
            joint_pair_paf,_ = model_utils.calculate_paf_mask(img, joint_pairs[i], keypoints, keypoint_type_to_idx, limb_width)
            ax.imshow(img)
            ax.imshow(joint_pair_paf.transpose(), 'jet', interpolation='none', alpha=0.5)
    plt.tight_layout()

def plot_heat_maps_from_annotations(img, anns, n_items=17, figsize=(16,12), sigma=7):
    fig, axes = plt.subplots(5, 4, figsize=figsize)
    img = np.array(img)
    fliped_img = img.transpose((1,0,2))
    kps = model_utils.get_keypoints_from_annotations(anns)
    
    for i,ax in enumerate(axes.flat):
        hide_subplot_axes(ax)
        if(i<17):
            joint_type = idx_to_keypoint_type[i]
            ax.text(10,10, joint_type, va='top', color="white", fontsize=12)
            mask,_ = model_utils.calculate_heatmap(img, i, kps, sigma)
            ax.imshow(img)
            ax.imshow(mask.transpose(), 'jet', interpolation='none', alpha=0.5)
    plt.tight_layout()

def plot_heatmaps(img, masks, idx_to_keypoint_type=idx_to_keypoint_type, figsize=(16,12)):
    fig, axes = plt.subplots(5, 4, figsize=figsize)
    
    for i,ax in enumerate(axes.flat):
        hide_subplot_axes(ax)
        if(i<17):
            joint_type = idx_to_keypoint_type[i]
            ax.text(10,10, joint_type, va='top', color="white", fontsize=12)
            ax.imshow(img)
            ax.imshow(masks[i], 'jet', interpolation='none', alpha=0.5)
        if(i==17):
            joint_type = "background"
            ax.text(10,10, joint_type, va='top', color="white", fontsize=12)
            ax.imshow(img)
            ax.imshow(masks[i], 'jet', interpolation='none', alpha=0.5)

    plt.tight_layout()

def plot_pafs(img, pafs, joint_pairs=part_pairs, figsize=(16,12)):
    fig, axes = plt.subplots(5, 4, figsize=figsize)
    
    for i,ax in enumerate(axes.flat):
        hide_subplot_axes(ax)
        if(i<19):
            ax.text(10,10, joint_pairs[i][0]+'->'+joint_pairs[i][1], va='top', color="white", fontsize=12)
            ax.imshow(img)
            mask = np.logical_or(pafs[2*i], pafs[(2*i) + 1]).astype(int)
            ax.imshow(mask, 'jet', interpolation='none', alpha=0.7)
    plt.tight_layout()
