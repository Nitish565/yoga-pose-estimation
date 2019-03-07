# yoga-pose-estimation
Model to classify yoga pose type and estimate joint positions of a person from an image

- Collect and clean image dataset of poses(107 asanas) <b>(Done)</b>
- Classifier model <b>(Done)</b>
- DS and classifier at: https://dopelemon.me/yoga-poses.html
![alt text](https://dopelemon.me/images/yoga-poses/pose-sample.png)


Pose Estimation part:
![alt text](http://cocodataset.org/images/keypoints-splash-big.png)
- Get join estimation model, training params from OpenPose <b>(Done)</b>
- OpenPose paper : https://arxiv.org/abs/1812.08008
- Earlier paper : https://arxiv.org/abs/1611.08050

- Download Microsoft's COCO dataset for labeled keypoints. (17 Joints and limb connections) <b>(Done)</b>
- COCO : http://cocodataset.org/#download 
- Write code to calculate ground truth joint Heatmaps and PAF's (Part affinity fields) from labeled keypoints <b>(Done)</b>
- Write Dataset class, custom transforms and vizualizations plots <b>(Done)</b>
- Write the NN Model to be trained in stages, implement from original paper <b>(Done)</b>

To-do:
- Write Loss calculation part
- Train model 
- Write the inference logic, nms to connect joints from pafs at test time to get final result

Sample Training Image:
![alt text](https://github.com/DhruvJawalkar/yoga-pose-estimation/blob/master/COCO/sample-trn-img.png)

PAFs:
![alt text](https://github.com/DhruvJawalkar/yoga-pose-estimation/blob/master/COCO/trn-img-paf-vectors.png)

Heatmaps:
![alt text](https://github.com/DhruvJawalkar/yoga-pose-estimation/blob/master/COCO/trn-img-heatmaps.png)
