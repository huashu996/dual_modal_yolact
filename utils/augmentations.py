import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
from math import sqrt

from data import cfg
from data import MEANS1, STD1
from data import MEANS2, STD2


class Compose(object):
    """Composes several augmentations together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img1, img2, masks=None, boxes=None, labels=None):
        for t in self.transforms:
            img1, img2, masks, boxes, labels = t(img1, img2, masks, boxes, labels)
        return img1, img2, masks, boxes, labels


class ConvertFromInts(object):
    def __call__(self, image1, image2, masks=None, boxes=None, labels=None):
        return image1.astype(np.float32), image2.astype(np.float32), masks, boxes, labels


class Resize(object):
    """ If preserve_aspect_ratio is true, this resizes to an approximate area of max_size * max_size """

    @staticmethod
    def calc_size_preserve_ar(img_w, img_h, max_size):
        """ I mathed this one out on the piece of paper. Resulting width*height = approx max_size^2 """
        ratio = sqrt(img_w / img_h)
        w = max_size * ratio
        h = max_size / ratio
        return int(w), int(h)

    def __init__(self, resize_gt=True):
        self.resize_gt = resize_gt
        self.max_size = cfg.max_size
        self.preserve_aspect_ratio = cfg.preserve_aspect_ratio

    def __call__(self, image1, image2, masks, boxes, labels=None):
        img_h, img_w, _ = image1.shape
        
        if self.preserve_aspect_ratio:
            width, height = Resize.calc_size_preserve_ar(img_w, img_h, self.max_size)
        else:
            width, height = self.max_size, self.max_size

        image1 = cv2.resize(image1, (width, height))
        image2 = cv2.resize(image2, (width, height))

        if self.resize_gt:
            # Act like each object is a color channel
            masks = masks.transpose((1, 2, 0))
            masks = cv2.resize(masks, (width, height))
            
            # OpenCV resizes a (w,h,1) array to (s,s), so fix that
            if len(masks.shape) == 2:
                masks = np.expand_dims(masks, 0)
            else:
                masks = masks.transpose((2, 0, 1))

            # Scale bounding boxes (which are currently absolute coordinates)
            boxes[:, [0, 2]] *= (width  / img_w)
            boxes[:, [1, 3]] *= (height / img_h)

        # Discard boxes that are smaller than we'd like
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        keep = (w > cfg.discard_box_width) * (h > cfg.discard_box_height)
        masks = masks[keep]
        boxes = boxes[keep]
        labels['labels'] = labels['labels'][keep]
        labels['num_crowds'] = (labels['labels'] < 0).sum()

        return image1, image2, masks, boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image1, image2, masks=None, boxes=None, labels=None):
        height, width, channels = image1.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image1, image2, masks, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image1, image2, masks=None, boxes=None, labels=None):
        # Only for image1
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image1 += delta
        return image1, image2, masks, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image1, image2, masks=None, boxes=None, labels=None):
        # Only for image1
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image1 *= alpha
        return image1, image2, masks, boxes, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image1, image2, masks=None, boxes=None, labels=None):
        # Only for image1
        if self.current == 'BGR' and self.transform == 'HSV':
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image1, image2, masks, boxes, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image1, image2, masks=None, boxes=None, labels=None):
        # Only for image1
        if random.randint(2):
            image1[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image1, image2, masks, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image1, image2, masks=None, boxes=None, labels=None):
        # Only for image1
        if random.randint(2):
            image1[:, :, 0] += random.uniform(-self.delta, self.delta)
            image1[:, :, 0][image1[:, :, 0] > 360.0] -= 360.0
            image1[:, :, 0][image1[:, :, 0] < 0.0] += 360.0
        return image1, image2, masks, boxes, labels


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A, B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A, B]
    union = area_a + area_b - inter
    return inter / union  # [A, B]


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(current='BGR', transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()

    def __call__(self, image1, image2, masks, boxes, labels):
        im1 = image1.copy()
        im2 = image2.copy()
        im1, im2, masks, boxes, labels = self.rand_brightness(im1, im2, masks, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im1, im2, masks, boxes, labels = distort(im1, im2, masks, boxes, labels)
        return im1, im2, masks, boxes, labels


class Expand(object):
    def __init__(self, mean1, mean2):
        self.mean1 = mean1
        self.mean2 = mean2

    def __call__(self, image1, image2, masks, boxes, labels):
        if random.randint(2):
            return image1, image2, masks, boxes, labels

        height, width, depth = image1.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image1 = np.zeros((int(height*ratio), int(width*ratio), depth), dtype=image1.dtype)
        expand_image1[:, :, :] = self.mean1
        expand_image1[int(top):int(top + height), int(left):int(left + width)] = image1
        image1 = expand_image1
        
        expand_image2 = np.zeros((int(height*ratio), int(width*ratio), depth), dtype=image2.dtype)
        expand_image2[:, :, :] = self.mean2
        expand_image2[int(top):int(top + height), int(left):int(left + width)] = image2
        image2 = expand_image2

        expand_masks = np.zeros(
            (masks.shape[0], int(height*ratio), int(width*ratio)),
            dtype=masks.dtype)
        expand_masks[:,int(top):int(top + height),
                       int(left):int(left + width)] = masks
        masks = expand_masks

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image1, image2, masks, boxes, labels


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image1, image2, masks, boxes=None, labels=None):
        height, width, _ = image1.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image1, image2, masks, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image1 = image1
                current_image2 = image2

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # This piece of code is bugged and does nothing:
                # https://github.com/amdegroot/ssd.pytorch/issues/68
                #
                # However, when I fixed it with overlap.max() < min_iou,
                # it cut the mAP in half (after 8k iterations). So it stays.
                #
                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image1 = current_image1[rect[1]:rect[3], rect[0]:rect[2], :]
                current_image2 = current_image2[rect[1]:rect[3], rect[0]:rect[2], :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # [0 ... 0 for num_gt and then 1 ... 1 for num_crowds]
                num_crowds = labels['num_crowds']
                crowd_mask = np.zeros(mask.shape, dtype=np.int32)

                if num_crowds > 0:
                    crowd_mask[-num_crowds:] = 1

                # have any valid boxes? try again if not
                # Also make sure you have at least one regular gt
                if not mask.any() or np.sum(1-crowd_mask[mask]) == 0:
                    continue

                # take only the matching gt masks
                current_masks = masks[mask, :, :].copy()

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                labels['labels'] = labels['labels'][mask]
                current_labels = labels

                # We now might have fewer crowd annotations
                if num_crowds > 0:
                    labels['num_crowds'] = np.sum(crowd_mask[mask])

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                # crop the current masks to the same dimensions as the image
                current_masks = current_masks[:, rect[1]:rect[3], rect[0]:rect[2]]

                return current_image1, current_image2, current_masks, current_boxes, current_labels


class RandomMirror(object):
    def __call__(self, image1, image2, masks, boxes, labels):
        _, width, _ = image1.shape
        if random.randint(2):
            image1 = image1[:, ::-1]
            image2 = image2[:, ::-1]
            masks = masks[:, :, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image1, image2, masks, boxes, labels


class Pad(object):
    """
    Pads the image to the input width and height, filling the
    background with mean and putting the image in the top-left.

    Note: this expects im_w <= width and im_h <= height
    """
    def __init__(self, width, height, mean1=MEANS1, mean2=MEANS2, pad_gt=True):
        self.mean1 = mean1
        self.mean2 = mean2
        self.width = width
        self.height = height
        self.pad_gt = pad_gt

    def __call__(self, image1, image2, masks, boxes=None, labels=None):
        im_h, im_w, depth = image1.shape

        expand_image1 = np.zeros((self.height, self.width, depth), dtype=image1.dtype)
        expand_image1[:, :, :] = self.mean1
        expand_image1[:im_h, :im_w] = image1
        
        expand_image2 = np.zeros((self.height, self.width, depth), dtype=image2.dtype)
        expand_image2[:, :, :] = self.mean2
        expand_image2[:im_h, :im_w] = image2

        if self.pad_gt:
            expand_masks = np.zeros((masks.shape[0], self.height, self.width), dtype=masks.dtype)
            expand_masks[:, :im_h, :im_w] = masks
            masks = expand_masks

        return expand_image1, expand_image2, masks, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image1, image2, masks=None, boxes=None, labels=None):
        height, width, channels = image1.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image1, image2, masks, boxes, labels


class PrepareMasks(object):
    """
    Prepares the gt masks for use_gt_bboxes by cropping with the gt box
    and downsampling the resulting mask to mask_size, mask_size. This
    function doesn't do anything if cfg.use_gt_bboxes is False.
    """

    def __init__(self, mask_size, use_gt_bboxes):
        self.mask_size = mask_size
        self.use_gt_bboxes = use_gt_bboxes

    def __call__(self, image1, image2, masks, boxes, labels=None):
        if not self.use_gt_bboxes:
            return image1, image2, masks, boxes, labels
        
        height, width, _ = image1.shape

        new_masks = np.zeros((masks.shape[0], self.mask_size ** 2))

        for i in range(len(masks)):
            x1, y1, x2, y2 = boxes[i, :]
            x1 *= width
            x2 *= width
            y1 *= height
            y2 *= height
            x1, y1, x2, y2 = (int(x1), int(y1), int(x2), int(y2))

            # +1 So that if y1=10.6 and y2=10.9 we still have a bounding box
            cropped_mask = masks[i, y1:(y2+1), x1:(x2+1)]
            scaled_mask = cv2.resize(cropped_mask, (self.mask_size, self.mask_size))

            new_masks[i, :] = scaled_mask.reshape(1, -1)
        
        # Binarize
        new_masks[new_masks >  0.5] = 1
        new_masks[new_masks <= 0.5] = 0

        return image1, image2, new_masks, boxes, labels


class BackboneTransform(object):
    """
    Transforms a BGR image made of floats in the range [0, 255] to whatever
    input the current backbone network needs.

    transform is a transform config object (see config.py).
    in_channel_order is probably 'BGR' but you do you, kid.
    """
    def __init__(self, transform, mean1, mean2, std1, std2, in_channel_order):
        self.mean1 = np.array(mean1, dtype=np.float32)
        self.mean2 = np.array(mean2, dtype=np.float32)
        self.std1  = np.array(std1,  dtype=np.float32)
        self.std2  = np.array(std2,  dtype=np.float32)
        self.transform = transform

        # Here I use "Algorithms and Coding" to convert string permutations to numbers
        self.channel_map = {c: idx for idx, c in enumerate(in_channel_order)}
        self.channel_permutation = [self.channel_map[c] for c in transform.channel_order]

    def __call__(self, img1, img2, masks=None, boxes=None, labels=None):

        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)

        if self.transform.normalize:
            img1 = (img1 - self.mean1) / self.std1
            img2 = (img2 - self.mean2) / self.std2
        elif self.transform.subtract_means:
            img1 = (img1 - self.mean1)
            img2 = (img2 - self.mean2)
        elif self.transform.to_float:
            img1 = img1 / 255
            img2 = img2 / 255

        img1 = img1[:, :, self.channel_permutation]
        img2 = img2[:, :, self.channel_permutation]

        return img1.astype(np.float32), img2.astype(np.float32), masks, boxes, labels


class BaseTransform(object):
    """ Transorm to be used when evaluating. """

    def __init__(self, mean1=MEANS1, mean2=MEANS2, std1=STD1, std2=STD2):
        self.augment = Compose([
            ConvertFromInts(),
            Resize(resize_gt=False),
            BackboneTransform(cfg.backbone.transform, mean1, mean2, std1, std2, 'BGR')
        ])

    def __call__(self, img1, img2, masks=None, boxes=None, labels=None):
        return self.augment(img1, img2, masks, boxes, labels)


def do_nothing(img1=None, img2=None, masks=None, boxes=None, labels=None):
    return img1, img2, masks, boxes, labels


def enable_if(condition, obj):
    return obj if condition else do_nothing


class SSDAugmentation(object):
    """ Transform to be used when training. """

    def __init__(self, mean1=MEANS1, mean2=MEANS2, std1=STD1, std2=STD2):
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            enable_if(cfg.augment_photometric_distort, PhotometricDistort()),
            enable_if(cfg.augment_expand, Expand(mean1, mean2)),
            enable_if(cfg.augment_random_sample_crop, RandomSampleCrop()),
            enable_if(cfg.augment_random_mirror, RandomMirror()),
            Resize(),
            enable_if(not cfg.preserve_aspect_ratio, Pad(cfg.max_size, cfg.max_size, mean1, mean2)),
            ToPercentCoords(),
            PrepareMasks(cfg.mask_size, cfg.use_gt_bboxes),
            BackboneTransform(cfg.backbone.transform, mean1, mean2, std1, std2, 'BGR')
        ])

    def __call__(self, img1, img2, masks, boxes, labels):
        return self.augment(img1, img2, masks, boxes, labels)
