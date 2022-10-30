import cv2
import math
import numpy as np
import os.path as osp
import torch
import torch.utils.data as data
from basicsr.data import degradations as degradations
from basicsr.data.data_util import paired_paths_from_folder
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
import json
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation,
                                               normalize)


@DATASET_REGISTRY.register()
class TalkingHeadDataset(data.Dataset):
    def __init__(self, opt):
        super(TalkingHeadDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder = opt['dataroot_gt']
        self.mean = opt['mean']
        self.std = opt['std']
        # self.out_size = opt['out_size']
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'
        
        self.crop_components = opt.get('crop_components', False)  # facial components
        self.eye_enlarge_ratio = opt.get('eye_enlarge_ratio', 1)  # whether enlarge eye regions

        if self.crop_components:
            # load component list from a pre-process pth files
            # self.components_list = torch.load(opt.get('component_path'))
            with open(opt['component_path']) as f:
                self.components_list = json.load(f)

        # file client (lmdb io backend)
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = self.gt_folder
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError(f"'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
            with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            # disk backend: scan file list from a folder
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

    def get_component_coordinates(self, image_name, status, scale_w, scale_h):
        """Get facial component (left_eye, right_eye, mouth) coordinates from a pre-loaded pth file"""
        components_bbox = self.components_list[image_name]

        # if status[0]:  # hflip
        #     # exchange right and left eye
        #     tmp = components_bbox['left_eye']
        #     components_bbox['left_eye'] = components_bbox['right_eye']
        #     components_bbox['right_eye'] = tmp
        #     # modify the width coordinate
        #     components_bbox['left_eye'][0] = self.out_size - components_bbox['left_eye'][0]
        #     components_bbox['right_eye'][0] = self.out_size - components_bbox['right_eye'][0]
        #     components_bbox['mouth'][0] = self.out_size - components_bbox['mouth'][0]

        # get  
        locations = []
        for part in ['left_eye', 'right_eye', 'mouth']:
            # mean = components_bbox[part][0:2]
            # half_len = components_bbox[part][2]

            x,y = component[image_name][part][0:2]
            w,h = component[image_name][part][2:]

            x = x / scale_w
            w = w / scale_w
            y = y / scale_h
            h = h / scale_h

            if 'eye' in part:
                half_len *= self.eye_enlarge_ratio

            loc = np.array([(x-w), (y-h), (x+w), (y+h)])
            loc = torch.from_numpy(loc).float()
            locations.append(loc)
        return locations

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']


        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        ori_size = img_gt.shape[:2]
        img_gt = cv2.resize(img_gt, self.opt['resize'], interpolation=cv2.INTER_LINEAR)

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)
        img_lq = cv2.resize(img_lq, self.opt['resize'], interpolation=cv2.INTER_LINEAR)

        # get facial component coordinates
        if self.crop_components:
            image_name = gt_path.split('/')[-1]
            scale_w, scale_h = (ori_size[0]/ self.opt['resize'][0]), (ori_size[1]/ self.opt['resize'][1])
            locations = self.get_component_coordinates(image_name, status, scale_w, scale_h)
            loc_left_eye, loc_right_eye, loc_mouth = locations
        
        # augmentation for training
        # if self.opt['phase'] == 'train':
        #     gt_size = self.opt['gt_size']
        #     # random crop
        #     img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
        #     # flip, rotation
        #     img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])
        
        img_gt = cv2.resize(img_gt, self.opt['resize'], interpolation=cv2.INTER_LINEAR)
        img_lq = cv2.resize(img_lq, self.opt['resize'], interpolation=cv2.INTER_LINEAR)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        if self.crop_components:
            return_dict = {
                'lq': img_lq,
                'gt': img_gt,
                'gt_path': gt_path,
                'loc_left_eye': loc_left_eye,
                'loc_right_eye': loc_right_eye,
                'loc_mouth': loc_mouth
            }
            return return_dict
        else:
            return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)

