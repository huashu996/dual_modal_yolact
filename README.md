# dual_modal_yolact

Dual-Modal Yolact in PyTorch for instance segmentation in complex environment

## Acknowledgement
 - This repository references [dbolya](https://github.com/dbolya/yolact)'s work.

## Installation
 - Install PyTorch environment with Anaconda (Test with Ubuntu 16.04)
   ```
   conda create -n yolact-env python=3.6.9
   conda activate yolact-env
   conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch
   pip install cython
   pip install opencv-python pillow pycocotools matplotlib
   ```
 - Clone this repository
   ```
   git clone https://github.com/shangjie-li/dual_modal_yolact.git
   ```

## Dataset
 - Check the dual modal dataset
   ```
   python dataset_player.py
   ```
   
## Training
 - Train on dual modal dataset
   ```
   python train.py
   ```
   
## Evaluation
 - Evaluate on dual modal dataset (mAP: 19.61 for box & 19.10 for mask)
   ```
   python eval.py --trained_model=weights/kitti_dual/yolact_base_35_20000.pth
   python eval.py --trained_model=weights/kitti_dual/yolact_base_35_20000.pth --display
   ```
   
## Demo
 - Run on the image-pair with trained model
   ```
   python eval.py --trained_model=weights/kitti_dual/yolact_base_35_20000.pth --image1=./data/kitti_dual/images/000001.png --image2=./data/kitti_dual/lidar_ddm_jet/000001.png --display
   python eval.py --trained_model=weights/kitti_dual/yolact_base_35_20000.pth --image1=./data/kitti_dual/images/000001.png --image2=./data/kitti_dual/lidar_ddm_jet/000001.png --save_image=test.png
   ```
 - Run on the image-pairs with trained model
   ```
   python eval.py --trained_model=weights/kitti_dual/yolact_base_35_20000.pth --images1=./data/kitti_dual/images --images2=./data/kitti_dual/lidar_ddm_jet --display
   python eval.py --trained_model=weights/kitti_dual/yolact_base_35_20000.pth --images1=./data/kitti_dual/images --images2=./data/kitti_dual/lidar_ddm_jet --save_images=kitti_dual_output
   ```

