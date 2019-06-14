import numpy as np

used_kps = np.array([0,1,2,3,4,5,10,11,12,13,14,15])

idx_to_keypoint_type = {0: 'r_ankle', 1: 'r_knee', 2: 'r_hip', 3: 'l_hip', 4: 'l_knee', 5: 'l_ankle', 6: 'pelvis', 7: 'thorax', 8: 'upper_neck', 9: 'head_top', 10: 'r_wrist', 11: 'r_elbow', 12: 'r_shoulder', 13: 'l_shoulder', 14: 'l_elbow', 15: 'l_wrist'}

truncated_kps_idx_to_type = {0: 'r_ankle', 1: 'r_knee', 2: 'r_hip', 3: 'l_hip', 4: 'l_knee', 5: 'l_ankle', 6: 'r_wrist', 7: 'r_elbow', 8: 'r_shoulder', 9: 'l_shoulder', 10: 'l_elbow', 11: 'l_wrist'}

truncated_kps_type_to_idx = {'r_ankle': 0, 'r_knee': 1, 'r_hip': 2, 'l_hip': 3, 'l_knee': 4, 'l_ankle': 5, 'r_wrist': 6, 'r_elbow': 7, 'r_shoulder': 8, 'l_shoulder': 9, 'l_elbow': 10, 'l_wrist': 11}

part_pairs = [['l_ankle', 'l_knee'], ['l_knee', 'l_hip'], ['r_ankle', 'r_knee'], ['r_knee', 'r_hip'], ['l_hip', 'r_hip'], ['l_shoulder', 'l_hip'], ['r_shoulder', 'r_hip'], ['l_shoulder', 'r_shoulder'], ['l_shoulder', 'l_elbow'], ['r_shoulder', 'r_elbow'], ['l_elbow', 'l_wrist'], ['r_elbow', 'r_wrist'], ['l_shoulder', 'l_wrist'], ['r_shoulder', 'r_wrist'], ['l_hip', 'l_ankle'], ['r_hip', 'r_ankle']]

keypoint_labels = ['r_ankle', 'r_knee', 'r_hip', 'l_hip', 'l_knee', 'l_ankle', 'pelvis', 'thorax', 'upper_neck', 'head_top', 'r_wrist', 'r_elbow', 'r_shoulder', 'l_shoulder', 'l_elbow', 'l_wrist']

truncated_kps_labels = ['r_ankle', 'r_knee', 'r_hip', 'l_hip', 'l_knee', 'l_ankle', 'r_wrist', 'r_elbow', 'r_shoulder', 'l_shoulder', 'l_elbow', 'l_wrist']

SKELETON = np.array([[5, 4], [4, 3], [0, 1], [1, 2], [3, 2], [9, 3], [8, 2], [9, 8], [9, 10], [8, 7], [10, 11], [7, 6], [9, 11], [8, 6], [3, 5], [2, 0]])

KEYPOINT_ORDER = np.arange(0,12)
GAUSSIAN_25x25 = np.load('gaussian_25x25.npy')

