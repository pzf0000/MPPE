import os

from pycocotools.coco import COCO

from PosturalRecognition.train.entity import params
from PosturalRecognition.train.coco_data_loader import CocoDataLoader

coco_train = COCO(os.path.join(params['coco_dir'], 'annotations/person_keypoints_train2017.json'))
coco_val = COCO(os.path.join(params['coco_dir'], 'annotations/person_keypoints_val2017.json'))
train_loader = CocoDataLoader(coco_train, mode='train')
val_loader = CocoDataLoader(coco_val, mode='val', n_samples=10)

