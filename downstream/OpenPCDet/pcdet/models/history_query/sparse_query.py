import MinkowskiEngine as ME
import torch
import torch.nn as nn
import numpy as np

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, loss_utils
from .sparse_resunet import load_model

# set kernel size >> number of history runs
HISTORY_KERNEL_SIZE = 1000

class SparseResUQueryNet(nn.Module):
    def __init__(
        self,
        model_cfg,
        num_class,
        num_point_features,
        **kwargs
    ):
        super().__init__()
        history_backbone_config = model_cfg.history_backbone_config
        history_backbone = model_cfg.history_backbone
        simple_conv_kernel_size = model_cfg.simple_conv_kernel_size
        extra_conv = model_cfg.extra_conv
        self.mode = model_cfg.mode
        self.model_cfg = model_cfg
        self.num_class = num_class
        assert self.mode in ('update_point_features', 'update_voxel_features')

        self.final_feature_size = history_backbone_config.final_feature_size
        self.history_backbone = load_model(history_backbone)(
            1,
            history_backbone_config.final_feature_size,
            history_backbone_config,
            3
        )
        self.agg_type = model_cfg.get('agg_type', 'max_pool')
        self.preserve_feature = 3
        if model_cfg.get('preserve_feature', False):
            self.preserve_feature = num_point_features
        if self.agg_type == 'max_pool':
            self.pool = ME.MinkowskiMaxPooling(
                kernel_size=[HISTORY_KERNEL_SIZE, 1, 1, 1],
                stride=[HISTORY_KERNEL_SIZE, 1, 1, 1],
                dimension=4
            )

        self.current_conv = \
            ME.MinkowskiConvolution(
                in_channels=history_backbone_config.final_feature_size,
                out_channels=history_backbone_config.final_feature_size,
                kernel_size=simple_conv_kernel_size, stride=1,
                dimension=3, expand_coordinates=False)
        self.extra_conv = extra_conv
        if self.extra_conv:
            self.extra_conv = nn.Sequential(
                ME.MinkowskiConvolution(
                    in_channels=history_backbone_config.final_feature_size,
                    out_channels=128,
                    kernel_size=3, stride=1,
                    dimension=3),
                ME.MinkowskiReLU(),
                ME.MinkowskiConvolution(
                        in_channels=128,
                        out_channels=64,
                        kernel_size=3, stride=1,
                        dimension=3),
                ME.MinkowskiReLU(),
                ME.MinkowskiConvolution(
                    in_channels=64,
                    out_channels=history_backbone_config.final_feature_size,
                    kernel_size=3, stride=1,
                    dimension=3),
            )
        if self.model_cfg.get("LOSS_CONFIG", None) is not None:
            self.cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
            self.cls_layers = self.make_fc_layers(
                fc_cfg=self.model_cfg.CLS_FC,
                input_channels=history_backbone_config.final_feature_size,
                output_channels=num_class
            )

    @property
    def point_dim(self):
        return self.final_feature_size + self.preserve_feature

    @staticmethod
    def make_fc_layers(fc_cfg, input_channels, output_channels):
        fc_layers = []
        c_in = input_channels
        for k in range(0, fc_cfg.__len__()):
            fc_layers.extend([
                nn.Linear(c_in, fc_cfg[k], bias=False),
                nn.BatchNorm1d(fc_cfg[k]),
                nn.ReLU(),
            ])
            c_in = fc_cfg[k]
        fc_layers.append(nn.Linear(c_in, output_channels, bias=True))
        return nn.Sequential(*fc_layers)

    def forward(self, batch):
        history_tensor = ME.SparseTensor(
            features=batch["history_features"],
            coordinates=batch["history_coordinates"],
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
        )
        output = self.history_backbone(history_tensor)
        if self.agg_type == 'max_pool':
            batched_output = ME.SparseTensor(
                features=output.features,
                coordinates=torch.hstack(
                    (batch['history_batches'].view(-1, 1), output.coordinates)),
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            )
            pooled_output = self.pool(batched_output)
            # assert torch.all(pooled_output.coordinates[:, 1] == 0)
            pooled_output = ME.SparseTensor(
                features=pooled_output.features,
                coordinates=pooled_output.coordinates[:, [0, 2, 3, 4]],
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            )
            curr_coord = batch["current_scan_coordinates"]
            query_field = ME.TensorField(
                features=torch.ones(
                    (curr_coord.shape[0], 1), device=curr_coord.device),
                coordinates=curr_coord,
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                # !!! this is important to set coordinate_manager!!!
                # without this the following query will be wrong
                coordinate_manager=pooled_output.coordinate_manager
            )
            query_sparse = query_field.sparse()
            conv_at_current = self.current_conv(
                pooled_output,
                coordinates=query_sparse
            )
            # slice the features
            ofield = conv_at_current.slice(query_field).F
        elif self.agg_type == 'avg_pool':
            extended_curr_coord = batch["expanded_current_scan_coordinates"]
            extended_curr_batches = batch['expanded_current_scan_batches']
            extended_query_field = ME.TensorField(
                features=torch.ones(
                    (extended_curr_coord.shape[0], 1), device=extended_curr_coord.device),
                coordinates=extended_curr_coord,
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                # !!! this is important to set coordinate_manager!!!
                # without this the following query will be wrong
                coordinate_manager=output.coordinate_manager
            )
            extended_query_sparse = extended_query_field.sparse()
            extended_conv_at_current = self.current_conv(
                output,
                coordinates=extended_query_sparse
            )
            extended_field = extended_conv_at_current.slice(
                extended_query_field).F
            # loop over the batches
            batch_size = batch["current_scan_coordinates"][:, 0].max()
            ofield = []
            for b in range(batch_size + 1):
                batch_mask = extended_curr_batches == b
                n_points = (batch["current_scan_coordinates"][:, 0] == b).sum()
                batch_key = extended_field[batch_mask].view(
                    -1, n_points, self.final_feature_size)
                ofield.append(batch_key.mean(dim=0))
            ofield = torch.cat(ofield)
        else:
            raise NotImplementedError(self.agg_type)
        if self.extra_conv:
            conv_at_current = self.extra_conv(conv_at_current)

        if self.mode == 'update_point_features':
            if self.preserve_feature > 3:
                batch['points'] = torch.cat(
                    (batch['points'], ofield), dim=1)
            else:
                batch['points'] = torch.cat((batch['points'][:, :4], ofield), dim=1)
        elif self.mode == 'update_voxel_features':
            voxel_feature = batch['voxels']
            voxel_feature_new = torch.zeros(
                (*voxel_feature.shape[:-1], self.final_feature_size),
                device=voxel_feature.device)
            mask = batch['voxel_point_mask']
            voxel_feature_new.masked_scatter_(
                (mask > -1), ofield[mask[mask > -1].long()])
            if self.preserve_feature > 3:
                batch['voxels'] = torch.cat(
                    (voxel_feature, voxel_feature_new), dim=-1)
            else:
                batch['voxels'] = torch.cat(
                    (voxel_feature[..., :3], voxel_feature_new), dim=-1)
        else:
            raise NotImplementedError(self.mode)

        if self.model_cfg.get("LOSS_CONFIG", None) is not None:
            self.forward_ret_dict = self.assign_target(batch)
            self.forward_ret_dict['sp_query_point_cls_preds'] = self.cls_layers(
                ofield)

        return batch

    def assign_target(self, input_dict):
        """
        Args:
            input_dict:
                batch_size:
                points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
        """
        points = input_dict['points'][:, :4]
        gt_boxes = input_dict['gt_boxes']
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert points.shape.__len__(
        ) in [2], 'points.shape=%s' % str(points.shape)

        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])

        bs_idx = points[:, 0]
        point_cls_labels = points.new_zeros(points.shape[0]).long()

        for k in range(batch_size):
            bs_mask = (bs_idx == k)
            points_single = points[bs_mask][:, 1:4]
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_single.unsqueeze(
                    dim=0), gt_boxes[k:k + 1, :, 0:7].contiguous()
            ).long().squeeze(dim=0)
            box_fg_flag = (box_idxs_of_pts >= 0)

            extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_single.unsqueeze(
                    dim=0), extend_gt_boxes[k:k+1, :, 0:7].contiguous()
            ).long().squeeze(dim=0)
            fg_flag = box_fg_flag
            ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
            point_cls_labels_single[ignore_flag] = -1

            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]
            point_cls_labels_single[fg_flag] = 1 if self.num_class == 1 else gt_box_of_fg_points[:, -1].long()
            point_cls_labels[bs_mask] = point_cls_labels_single

        return {'sp_query_point_cls_labels': point_cls_labels}

    def get_loss(self, tb_dict=None):
        point_cls_labels = self.forward_ret_dict['sp_query_point_cls_labels'].view(-1)
        point_cls_preds = self.forward_ret_dict['sp_query_point_cls_preds'].view(
            -1, self.num_class)

        positives = (point_cls_labels > 0)
        negative_cls_weights = (point_cls_labels == 0) * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        pos_normalizer = positives.sum(dim=0).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        one_hot_targets = point_cls_preds.new_zeros(
            *list(point_cls_labels.shape), self.num_class + 1)
        one_hot_targets.scatter_(-1, (point_cls_labels *
                                 (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.cls_loss_func(
            point_cls_preds, one_hot_targets, weights=cls_weights)
        point_loss_cls = cls_loss_src.sum()

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_cls = point_loss_cls * loss_weights_dict['point_cls_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({
            'sp_point_loss_cls': point_loss_cls.item(),
        })
        return point_loss_cls, tb_dict
