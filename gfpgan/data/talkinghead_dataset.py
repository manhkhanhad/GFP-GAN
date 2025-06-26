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
            with open(opt['component_path']) as f:
                self.components_data = json.load(f)
            self.video_names = list(self.components_data.keys())

            self.paths = []
            self.components_list = {}
            
            for video_name in self.video_names:
                frame_names = self.components_data[video_name]
                for frame_name, component in frame_names.items():
                    lq_path = osp.join(self.lq_folder, video_name, f"{int(frame_name):03d}.png")
                    gt_path = osp.join(self.gt_folder, video_name, f"{int(frame_name):03d}.png")
                    if not osp.exists(lq_path) or not osp.exists(gt_path):
                        continue
                    component_key = f"{video_name}_{int(frame_name):03d}"
                    self.paths.append({'lq_path': lq_path, 'gt_path': gt_path, 'key': component_key})
                    self.components_list[component_key] = component
                    
            # self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)
            # # Check all paths are valid, if not, remove the path
            # breakpoint()
            # invalid_paths = 0
            # for path in self.paths:
            #     if not osp.exists(path['lq_path']) or not osp.exists(path['gt_path']):
            #         self.paths.remove(path)
            #         invalid_paths += 1
            # print(f"Removed {invalid_paths} invalid paths")


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
            if components_bbox[part] is None:
                locations.append([0,0,0,0])
                continue

            x,y = components_bbox[part][0:2]
            w,h = components_bbox[part][2:4]

            x = x / scale_w
            w = w / scale_w
            y = y / scale_h
            h = h / scale_h

            # if 'eye' in part:
            #     half_len *= self.eye_enlarge_ratio

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
        try:
            gt_path = self.paths[index]['gt_path']
            lq_path = self.paths[index]['lq_path']
            key = self.paths[index]['key']

            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gt = cv2.imread(gt_path).astype(np.float32) / 255.0
            ori_size = img_gt.shape[:2]
            img_gt = cv2.resize(img_gt, self.opt['resize'], interpolation=cv2.INTER_LINEAR)

            
            img_bytes = self.file_client.get(lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lq = cv2.resize(img_lq, self.opt['resize'], interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            print(f"Error loading gt image: {e}")
            print(f"Gt path: {gt_path}")
            print(f"Lq path: {lq_path}")
            return self.__getitem__(index + 1)
        

        # get facial component coordinates
        if self.crop_components:
            image_name = gt_path.split('/')[-1]
            scale_w, scale_h = (ori_size[0]/ self.opt['resize'][0]), (ori_size[1]/ self.opt['resize'][1])
            locations = self.get_component_coordinates(key, 0, scale_w, scale_h)
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
                'loc_left_eye': torch.tensor(loc_left_eye),
                'loc_right_eye': torch.tensor(loc_right_eye),
                'loc_mouth': torch.tensor(loc_mouth)
            }
            return return_dict
        else:
            return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)

