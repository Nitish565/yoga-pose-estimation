# yoga-pose-estimation
Model to classify yoga pose type and estimate joint positions of a person from an image

- Collect and clean image dataset of poses(107 asanas) <b>(Done)</b>
- Classifier model <b>(Done)</b>
- DS, pose classifier + joint estimation model at: https://dopelemon.me/yoga-poses.html
- Pose Estimation model currently borrowed from https://github.com/Hzzone/pytorch-openpose
![alt text](https://github.com/DhruvJawalkar/blog/blob/master/app/results/yoga-pose/res-7100780243.png)


Pose Estimation part:
![alt text](http://cocodataset.org/images/keypoints-splash-big.png)
- Get join estimation model, training params from OpenPose <b>(Done)</b>
- OpenPose paper : https://arxiv.org/abs/1812.08008
- Earlier paper : https://arxiv.org/abs/1611.08050

- Download Microsoft's COCO dataset for labeled keypoints. (17 Joints and 38 limb connections) <b>(Done)</b>
- COCO : http://cocodataset.org/#download 
- Write code to calculate ground truth joint Heatmaps and PAF's (Part affinity fields) from labeled keypoints <b>(Done)</b>
- Write Dataset class, custom transforms and vizualizations plots <b>(Done)</b>
- Write the NN Model to be trained in stages, implement from original paper <b>(Done)</b>
- Write Loss calculation part <b>(Done)</b>
- Train model (PAF Stages 1-4) <b>(Done)</b>
- Train model (Heatmaps Stages 1-2) <b>(Done)</b>

In Progress:
- Write the inference logic, nms to connect joints from prdicted pafs, heatmaps. 

Sample Training Image:
![alt text](https://github.com/DhruvJawalkar/yoga-pose-estimation/blob/master/COCO/sample-trn-img.png)

PAFs:
![alt text](https://github.com/DhruvJawalkar/yoga-pose-estimation/blob/master/COCO/trn-img-paf-vectors.png)

Heatmaps:
![alt text](https://github.com/DhruvJawalkar/yoga-pose-estimation/blob/master/COCO/trn-img-heatmaps.png)
