import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import random

try:
    import cv2
except ImportError:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

from data import COCODetection
from data import cfg, set_cfg, set_dataset
from data import MEANS1, MEANS2, STD1, STD2
from utils.augmentations import SSDAugmentation

def create_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    
    color = (r, g, b)
    return color

def draw_mask(img, mask, color):
    img_gpu = torch.from_numpy(img).cuda().float()
    img_gpu = img_gpu / 255.0
    
    mask = mask[:, :, None]
    color_tensor = torch.Tensor(color).to(img_gpu.device.index).float() / 255.
    alpha = 0.45
    
    mask_color = mask.repeat(1, 1, 3) * color_tensor * alpha
    inv_alph_mask = mask * (- alpha) + 1
    img_gpu = img_gpu * inv_alph_mask + mask_color
    img_numpy = (img_gpu * 255).byte().cpu().numpy()
    
    return img_numpy

def draw_annotation(img, mask, box, classname, color, score=None):
    font_face = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.4
    font_thickness, line_thickness = 1, 2
    
    x1, y1, x2, y2 = box[:]
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, line_thickness)
    
    img = draw_mask(img, mask, color)
    
    u, v = int(x1), int(y1)
    text_str = '%s: %.2f' % (classname, score) if score else classname
    text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
    
    if v - text_h - 4 < 0: v = text_h + 4
    cv2.rectangle(img, (u, v), (u + text_w, v - text_h - 4), color, -1)
    cv2.putText(img, text_str, (u, v - 3), font_face, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    
    return img

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if len(sys.argv) == 2:
            # Usage: python dataset_player.py kitti_dual
            dataset_name = sys.argv[1]
            set_dataset(dataset_name)
            print('Using dataset:')
            cfg.dataset.print()
            print()
        else:
            print('Error: Only one parameter (dataset_name) is supported.')
            exit()
    else:
        print('Using default dataset:')
        cfg.dataset.print()
        print()
    

    dataset = COCODetection(image_path1=cfg.dataset.train_images1,
                            image_path2=cfg.dataset.train_images2,
                            info_file=cfg.dataset.train_info,
                            transform=SSDAugmentation(),
                            dataset_name=cfg.dataset.name,
                            has_gt=cfg.dataset.has_gt)

    for i in range(len(dataset)):
        print('\n--------[%d/%d]--------' % (i + 1, len(dataset)))
        
        print('Original Data:')
        img_id_i, file_name, img1, img2, height, width = dataset.pull_image(i) # BGR
        img_id_a, masks, target, num_crowds = dataset.pull_anno(i, height, width)
        
        print(' image_id:', img_id_i, ' file_name:', file_name)
        print(' img1:', type(img1), img1.shape, ' h: %s w: %s' % (height, width), '\n', img1)
        print(' img2:', type(img2), img2.shape, ' h: %s w: %s' % (height, width), '\n', img2)
        print(' masks:', type(masks), masks.shape, '\n', masks)
        
        boxes = target[:, :4]
        labels = target[:, 4]
        
        print(' boxes:', type(boxes), boxes.shape,'\n', boxes)
        print(' labels:', type(labels), labels.shape,'\n', labels)
        print(' num_crowds:', type(num_crowds), '\n', num_crowds)
        
        img_annotated1 = img1.copy()
        img_annotated2 = img2.copy()
        scale = [width, height, width, height]
        
        for j in range(boxes.shape[0]):
            mask = torch.from_numpy(masks[j]).cuda().float()
            box = boxes[j] * scale
            label = int(labels[j])
            if label >= 0:
                classname = cfg.dataset.class_names[label]
            else:
                classname = 'crowd'
            
            color = create_random_color()
            img_annotated1 = draw_annotation(img_annotated1, mask, box, classname, color)
            img_annotated2 = draw_annotation(img_annotated2, mask, box, classname, color)
        cv2.imshow('Original Data 1', img_annotated1)
        cv2.imshow('Original Data 2', img_annotated2)
        
        
        print()
        print('Augmented Data:')
        img_aug, (gt_aug, masks_aug, num_crowds_aug) = dataset[i]
        _, height_aug, width_aug = img_aug.shape
        
        img_aug = torch.split(img_aug, [3, 3], dim=0)
        img_aug1 = img_aug[0]
        img_aug2 = img_aug[1]
        
        print(' img_aug1:', type(img_aug1), img_aug1.shape, ' h: %s w: %s' % (height_aug, width_aug), '\n', img_aug1)
        print(' img_aug2:', type(img_aug2), img_aug2.shape, ' h: %s w: %s' % (height_aug, width_aug), '\n', img_aug2)
        print(' masks_aug:', type(masks_aug), masks_aug.shape, '\n', masks_aug)
        
        boxes_aug = gt_aug[:, :4]
        labels_aug = gt_aug[:, 4]
        
        print(' boxes_aug:', type(boxes_aug), boxes_aug.shape, '\n', boxes_aug)
        print(' labels_aug:', type(labels_aug), labels_aug.shape,'\n', labels_aug)
        print(' num_crowds_aug:', type(num_crowds_aug), '\n', num_crowds_aug)
        
        img_aug_numpy1 = img_aug1.cpu().numpy().astype(np.float32).transpose((1, 2, 0))
        img_aug_numpy1 = ((img_aug_numpy1 * STD1) + MEANS1).astype(np.uint8)[:, :, ::-1] # BGR
        img_aug_annotated1 = img_aug_numpy1.copy()
        
        img_aug_numpy2 = img_aug2.cpu().numpy().astype(np.float32).transpose((1, 2, 0))
        img_aug_numpy2 = ((img_aug_numpy2 * STD2) + MEANS2).astype(np.uint8)[:, :, ::-1] # BGR
        img_aug_annotated2 = img_aug_numpy2.copy()
        
        scale_aug = [width_aug, height_aug, width_aug, height_aug]
        
        for j in range(boxes_aug.shape[0]):
            mask = torch.from_numpy(masks_aug[j]).cuda().float()
            box = boxes_aug[j] * scale_aug
            label = int(labels_aug[j])
            if label >= 0:
                classname = cfg.dataset.class_names[label]
            else:
                classname = 'crowd'
            
            color = create_random_color()
            img_aug_annotated1 = draw_annotation(img_aug_annotated1, mask, box, classname, color)
            img_aug_annotated2 = draw_annotation(img_aug_annotated2, mask, box, classname, color)
        cv2.imshow('Augmented Data 1', img_aug_annotated1)
        cv2.imshow('Augmented Data 2', img_aug_annotated2)
        
        # press 'Esc' to shut down, and every key else to continue
        key = cv2.waitKey(0)
        if key == 27:
            break
        else:
            continue
