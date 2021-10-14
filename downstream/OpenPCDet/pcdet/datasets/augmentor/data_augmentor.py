from functools import partial
from kornia.geometry.transform.affwarp import scale

import numpy as np
import torch
import MinkowskiEngine as ME

from ...utils import common_utils
from . import augmentor_utils, database_sampler


class DataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger

        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        augmentation_names = [cur_cfg.NAME for cur_cfg in aug_config_list]
        # if 'point_quantize' in augmentation_names:
        #     if 'gt_sampling' in augmentation_names:
        #         assert augmentation_names.index(
        #             'gt_sampling') < augmentation_names.index('point_quantize')
        #     for i in range(len(augmentation_names)):
        #         if 'random' in augmentation_names[i]:
        #             assert i > augmentation_names.index('point_quantize')
        self.augmentation_names = augmentation_names

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def gt_sampling(self, config=None):
        db_sampler = database_sampler.DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def point_quantize(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.point_quantize, config=config)
        current_scan = data_dict['points']
        if config.get("RANDOM_ERROR_TEST", -1.0) > 0:
            _cur_scan = current_scan[:, :3].copy()
            error_vector = np.random.multivariate_normal(
                mean=(0,0,0),
                cov=np.eye(3), size=(1, ))
            error_vector /= np.linalg.norm(error_vector)
            error_vector *= np.random.normal(scale=config.RANDOM_ERROR_TEST)
            # print(np.linalg.norm(error_vector))
            _cur_scan += error_vector
            data_dict['current_scan_coordinates'] = ( _cur_scan
                 / config['VOXEL_SIZE']).astype(np.int32)
        else:
            data_dict['current_scan_coordinates'] = (
                current_scan[:, :3].copy() / config['VOXEL_SIZE']).astype(np.int32)
        if data_dict.get('history_scans', False):
            data_dict["history_coordinates"] = [ME.utils.sparse_quantize(
                x / config['VOXEL_SIZE']) for x in data_dict['history_scans']]
            data_dict['history_features'] = [torch.ones((len(x), 1))
                                for x in data_dict["history_coordinates"]]
            data_dict.pop('history_scans')
        return data_dict

    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        history_scans = data_dict.get('history_scans', None)
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            gt_boxes, points, history_scans = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                gt_boxes, points, history_scans
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        if history_scans is not None:
            data_dict['history_scans'] = history_scans
        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        history_scans = data_dict.get('history_scans', None)
        gt_boxes, points, history_scans = augmentor_utils.global_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range, history_scans=history_scans
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        if history_scans is not None:
            data_dict['history_scans'] = history_scans
        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        history_scans = data_dict.get('history_scans', None)
        gt_boxes, points, history_scans = augmentor_utils.global_scaling(
            data_dict['gt_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE'], history_scans=history_scans
        )
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        if history_scans is not None:
            data_dict['history_scans'] = history_scans
        return data_dict

    def random_image_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_image_flip, config=config)
        images = data_dict["images"]
        depth_maps = data_dict["depth_maps"]
        gt_boxes = data_dict['gt_boxes']
        gt_boxes2d = data_dict["gt_boxes2d"]
        calib = data_dict["calib"]
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['horizontal']
            images, depth_maps, gt_boxes = getattr(augmentor_utils, 'random_image_flip_%s' % cur_axis)(
                images, depth_maps, gt_boxes, calib,
            )

        data_dict['images'] = images
        data_dict['depth_maps'] = depth_maps
        data_dict['gt_boxes'] = gt_boxes
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)

        data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
            data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
        )
        if 'calib' in data_dict:
            data_dict.pop('calib')
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')
        if 'gt_boxes_mask' in data_dict:
            gt_boxes_mask = data_dict['gt_boxes_mask']
            data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
            data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]
            if 'gt_boxes2d' in data_dict:
                data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][gt_boxes_mask]

            data_dict.pop('gt_boxes_mask')
        return data_dict
