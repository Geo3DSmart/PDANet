import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import box_coder_utils, box_utils, loss_utils, common_utils,nn_distance
from .point_head_template import PointHeadTemplate
from ...ops.roiaware_pool3d import roiaware_pool3d_utils


from ...ops.pointnet2.pointnet2_batch import pointnet2_utils
from ...ops.pointnet2.pointnet2_batch import cd_loss

from .cluster_contrastloss import clusterContrastLoss

class IASSD_Head(PointHeadTemplate):
    """
    A simple point-based detect head, which are used for IA-SSD.
    """
    def __init__(self, num_class, input_channels, model_cfg, predict_boxes_when_training=False, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.predict_boxes_when_training = predict_boxes_when_training

        target_cfg = self.model_cfg.TARGET_CONFIG
        self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(
            **target_cfg.BOX_CODER_CONFIG
        )
        detector_dim = self.model_cfg.get('INPUT_DIM', input_channels) # for spec input_channel
        self.cls_center_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=detector_dim,
            output_channels=num_class
        )
        self.box_center_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.REG_FC,
            input_channels=detector_dim,
            output_channels=self.box_coder.code_size
        )
        
        self.box_iou3d_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.IOU_FC,
            input_channels=detector_dim,
            output_channels=1
        ) if self.model_cfg.get('IOU_FC', None) is not None else None

        # self.init_weights(weight_init='xavier')
        self.nce = clusterContrastLoss(ignore_label=-1, k=40, mu=0.9999)
        self.jshu =0


    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def build_losses(self, losses_cfg):
        # classification loss
        if losses_cfg.LOSS_CLS.startswith('WeightedBinaryCrossEntropy'):
            self.add_module(
                'cls_loss_func',
                loss_utils.WeightedBinaryCrossEntropyLoss()
            )
        elif losses_cfg.LOSS_CLS.startswith('WeightedCrossEntropy'):
            self.add_module(
                'cls_loss_func',
                loss_utils.WeightedClassificationLoss()
            )
        elif losses_cfg.LOSS_CLS.startswith('FocalLoss'):
            self.add_module(
                'cls_loss_func',
                loss_utils.SigmoidFocalClassificationLoss(
                    **losses_cfg.get('LOSS_CLS_CONFIG', {})
                )
            )
        else:
            raise NotImplementedError

        # regression loss
        if losses_cfg.LOSS_REG == 'WeightedSmoothL1Loss':
            self.add_module(
                'reg_loss_func',
                loss_utils.WeightedSmoothL1Loss(
                    code_weights=losses_cfg.LOSS_WEIGHTS.get('code_weights', None),
                    **losses_cfg.get('LOSS_REG_CONFIG', {})
                )
            )
        elif losses_cfg.LOSS_REG == 'WeightedL1Loss':
            self.add_module(
                'reg_loss_func',
                loss_utils.WeightedL1Loss(
                    code_weights=losses_cfg.LOSS_WEIGHTS.get('code_weights', None)
                )
            )
        else:
            raise NotImplementedError

        # instance-aware loss
        if losses_cfg.get('LOSS_INS', None) is not None:
            if losses_cfg.LOSS_INS.startswith('WeightedBinaryCrossEntropy'):
                self.add_module(
                    'ins_loss_func',
                    loss_utils.WeightedBinaryCrossEntropyLoss()
                )
            elif losses_cfg.LOSS_INS.startswith('WeightedCrossEntropy'):
                self.add_module(
                    'ins_loss_func',
                    loss_utils.WeightedClassificationLoss()
                )
            elif losses_cfg.LOSS_INS.startswith('FocalLoss'):
                self.add_module(
                    'ins_loss_func',
                    loss_utils.SigmoidFocalClassificationLoss(
                        **losses_cfg.get('LOSS_CLS_CONFIG', {})
                    )
                )
            else:
                raise NotImplementedError
    #  set_ignore_flag=True, use_ex_gt_assign= False
    def assign_stack_targets_IASSD(self, points, gt_boxes, extend_gt_boxes=None, weighted_labels=False,
                             ret_box_labels=False, ret_offset_labels=True,
                             set_ignore_flag=True, use_ball_constraint=False, central_radius=2.0,
                             use_query_assign=False, central_radii=2.0, use_ex_gt_assign=False, fg_pc_ignore=False,
                             binary_label=False):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
            extend_gt_boxes: [B, M, 8]
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_box_labels: (N1 + N2 + N3 + ..., code_size)

        """
        assert len(points.shape) == 2 and points.shape[1] == 4, 'points.shape=%s' % str(points.shape)
        assert len(gt_boxes.shape) == 3 and gt_boxes.shape[2] == 8, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3 and extend_gt_boxes.shape[2] == 8, \
            'extend_gt_boxes.shape=%s' % str(extend_gt_boxes.shape)
        batch_size = gt_boxes.shape[0]
        bs_idx = points[:, 0]
        # Init Class ，All 0； shape （batch * 16384）
        point_cls_labels = points.new_zeros(points.shape[0]).long()
        # Fault: 0； shape （batch * 16384, 8）
        point_box_labels = gt_boxes.new_zeros((points.shape[0], 8)) if ret_box_labels else None
        box_idxs_labels = points.new_zeros(points.shape[0]).long() 
        gt_boxes_of_fg_points = []
        gt_box_of_points = gt_boxes.new_zeros((points.shape[0], 8))

        for k in range(batch_size):      
            
            bs_mask = (bs_idx == k)
            # point shape   (16384, 3)
            points_single = points[bs_mask][:, 1:4]
            # All 0    (16384, )
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
            #  box_idxs_of_pts
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_single.unsqueeze(dim=0), gt_boxes[k:k + 1, :, 0:7].contiguous()
            ).long().squeeze(dim=0)
             #  # mask Forground/Background points
            box_fg_flag = (box_idxs_of_pts >= 0) 
            if use_query_assign: ## False
                centers = gt_boxes[k:k + 1, :, 0:3]
                """
                points_single : (16384, 3) --> (1, 16384, 3)
                gt_boxes : (batch, num_of_GTs, 8)
                box_idxs_of_pts : (16384, )，
                """
                # , box_idxs_of_pts

                query_idxs_of_pts = roiaware_pool3d_utils.points_in_ball_query_gpu(
                    points_single.unsqueeze(dim=0), centers.contiguous(), central_radii
                    ).long().squeeze(dim=0) 
                query_fg_flag = (query_idxs_of_pts >= 0)
                if fg_pc_ignore:
                    fg_flag = query_fg_flag ^ box_fg_flag 
                    extend_box_idxs_of_pts[box_idxs_of_pts!=-1] = -1
                    box_idxs_of_pts = extend_box_idxs_of_pts
                else:
                    fg_flag = query_fg_flag
                    box_idxs_of_pts = query_idxs_of_pts
            elif use_ex_gt_assign: ## False
                # GTbox_enlarge中
                extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0), extend_gt_boxes[k:k+1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                # 在 GTbox_enlarge 的点
                extend_fg_flag = (extend_box_idxs_of_pts >= 0)
                # GTbox_enlarge 
                extend_box_idxs_of_pts[box_fg_flag] = box_idxs_of_pts[box_fg_flag] #instance points should keep unchanged

                if fg_pc_ignore: # False
                    fg_flag = extend_fg_flag ^ box_fg_flag
                    extend_box_idxs_of_pts[box_idxs_of_pts!=-1] = -1
                    box_idxs_of_pts = extend_box_idxs_of_pts
                else:
                    fg_flag = extend_fg_flag  # 
                    box_idxs_of_pts = extend_box_idxs_of_pts # 
            #  True               
            elif set_ignore_flag: 
                # 
                extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0), extend_gt_boxes[k:k+1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
               
                fg_flag = box_fg_flag 
              
                ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
               
                point_cls_labels_single[ignore_flag] = -1 

            elif use_ball_constraint: 
                box_centers = gt_boxes[k][box_idxs_of_pts][:, 0:3].clone()
                box_centers[:, 2] += gt_boxes[k][box_idxs_of_pts][:, 5] / 2
                ball_flag = ((box_centers - points_single).norm(dim=1) < central_radius)
                fg_flag = box_fg_flag & ball_flag

            else:
                raise NotImplementedError
            
           
    
            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]   # 
       
            point_cls_labels_single[fg_flag] = 1 if self.num_class == 1 or binary_label else gt_box_of_fg_points[:, -1].long()
   
            point_cls_labels[bs_mask] = point_cls_labels_single

            bg_flag = (point_cls_labels_single == 0) # except ignore_id
            
            # box_bg_flag
            # 
            fg_flag = fg_flag ^ (fg_flag & bg_flag)  # 
            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]

            gt_boxes_of_fg_points.append(gt_box_of_fg_points)
            box_idxs_labels[bs_mask] = box_idxs_of_pts
            gt_box_of_points[bs_mask] = gt_boxes[k][box_idxs_of_pts]
     
            if ret_box_labels and gt_box_of_fg_points.shape[0] > 0:
          
                point_box_labels_single = point_box_labels.new_zeros((bs_mask.sum(), 8))
               
                fg_point_box_labels = self.box_coder.encode_torch(
                    gt_boxes=gt_box_of_fg_points[:, :-1], points=points_single[fg_flag],
                    gt_classes=gt_box_of_fg_points[:, -1].long()
                )
             
                # fg_point_box_labels: (num_of_GT_matched_by_point,8)
                # point_box_labels_single: (16384, 8）
                point_box_labels_single[fg_flag] = fg_point_box_labels
                point_box_labels[bs_mask] = point_box_labels_single


        gt_boxes_of_fg_points = torch.cat(gt_boxes_of_fg_points, dim=0)
        targets_dict = {
            # shape (batch * 16384)
            'point_cls_labels': point_cls_labels,
            # shape (batch * 16384) shape (batch * 16384, 8)
            'point_box_labels': point_box_labels,
            'gt_box_of_fg_points': gt_boxes_of_fg_points,
            'box_idxs_labels': box_idxs_labels,
            'gt_box_of_points': gt_box_of_points,
        }
        return targets_dict

    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                batch_size: int
                centers: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                centers_origin: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                encoder_coords: List of point_coords in SA
                gt_boxes (optional): (B, M, 8)
        Returns:
            target_dict:
            ...
        """
        target_cfg = self.model_cfg.TARGET_CONFIG
        gt_boxes = input_dict['gt_boxes']
        if gt_boxes.shape[-1] == 10:   #nscence
            gt_boxes = torch.cat((gt_boxes[..., 0:7], gt_boxes[..., -1:]), dim=-1)

        targets_dict_center = {}
        # assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        # assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)
        batch_size = input_dict['batch_size']      
        if target_cfg.get('EXTRA_WIDTH', False):  # multi class extension
            extend_gt = box_utils.enlarge_box3d_for_class(
                gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=target_cfg.EXTRA_WIDTH
            ).view(batch_size, -1, gt_boxes.shape[-1])
        else:
            extend_gt = gt_boxes

        extend_gt_boxes = box_utils.enlarge_box3d(
            extend_gt.view(-1, extend_gt.shape[-1]), extra_width=target_cfg.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        center_targets_dict = self.assign_stack_targets_IASSD(
            points=input_dict['centers'].detach(), 
            gt_boxes=extend_gt, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,  # set_ignore_flag:
            ret_box_labels=True
        )
        targets_dict_center['center_gt_box_of_fg_points'] = center_targets_dict['gt_box_of_fg_points']
        targets_dict_center['center_cls_labels'] = center_targets_dict['point_cls_labels']
        targets_dict_center['center_box_labels'] = center_targets_dict['point_box_labels'] #only center assign
        targets_dict_center['center_gt_box_of_points'] = center_targets_dict['gt_box_of_points']
        if target_cfg.get('INS_AWARE_ASSIGN', False):
            sa_ins_labels, sa_gt_box_of_fg_points, sa_xyz_coords, sa_gt_box_of_points, sa_box_idxs_labels = [],[],[],[],[]
            sa_ins_preds = input_dict['sa_ins_preds']  # [list] 

            # encoder_sample_list_id = input_dict['sample_list_id']

            get_origin_class = []

            for i in range(0, 1): # valid when i = 1,2 for IA-SSD
                # if sa_ins_preds[i].__len__() == 0:
                #     continue
                sa_xyz = input_dict['encoder_coords'][i]  #
                if i == 0:
                    extend_gt_boxes = box_utils.enlarge_box3d(
                        gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=[0.5, 0.5, 0.5]  #[0.2, 0.2, 0.2]  
                    ).view(batch_size, -1, gt_boxes.shape[-1])    
                             
                    sa_targets_dict = self.assign_stack_targets_IASSD(
                        points=sa_xyz.view(-1,sa_xyz.shape[-1]).detach(), gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
                        set_ignore_flag=True, use_ex_gt_assign= False      # # set_ignore_flag: True  
                    )
     
                get_origin_class.append(sa_targets_dict['point_cls_labels'])



            for i in range(1, len(sa_ins_preds)): # valid when i = 1,2 for IA-SSD
                # if sa_ins_preds[i].__len__() == 0:
                #     continue
                sa_xyz = input_dict['encoder_coords'][i]  # [list]
                if i == 1:
                    extend_gt_boxes = box_utils.enlarge_box3d(
                        gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=[0.5, 0.5, 0.5]  #[0.2, 0.2, 0.2] 
                    ).view(batch_size, -1, gt_boxes.shape[-1])    
                             
                    sa_targets_dict = self.assign_stack_targets_IASSD(
                        points=sa_xyz.view(-1,sa_xyz.shape[-1]).detach(), gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
                        set_ignore_flag=True, use_ex_gt_assign= False      # # set_ignore_flag:
                    )
                if i >= 2:
                # if False:
                    extend_gt_boxes = box_utils.enlarge_box3d(
                        gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=[0.5, 0.5, 0.5]
                    ).view(batch_size, -1, gt_boxes.shape[-1]) 
                                
                    sa_targets_dict = self.assign_stack_targets_IASSD(
                        points=sa_xyz.view(-1,sa_xyz.shape[-1]).detach(), gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
                        set_ignore_flag=False, use_ex_gt_assign= True 
                    )
                # else:
                #     extend_gt_boxes = box_utils.enlarge_box3d(
                #         gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=[0.5, 0.5, 0.5]
                #     ).view(batch_size, -1, gt_boxes.shape[-1]) 
                #     sa_targets_dict = self.assign_stack_targets_IASSD(
                #         points=sa_xyz.view(-1,sa_xyz.shape[-1]).detach(), gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
                #         set_ignore_flag=False, use_ex_gt_assign= True 
                #     )
                sa_xyz_coords.append(sa_xyz)
                sa_ins_labels.append(sa_targets_dict['point_cls_labels'])    
                sa_gt_box_of_fg_points.append(sa_targets_dict['gt_box_of_fg_points'])  
                sa_gt_box_of_points.append(sa_targets_dict['gt_box_of_points'])
                sa_box_idxs_labels.append(sa_targets_dict['box_idxs_labels'])                
                
            targets_dict_center['sa_ins_labels'] = sa_ins_labels
            targets_dict_center['get_origin_class_label'] = get_origin_class
            
            targets_dict_center['sa_gt_box_of_fg_points'] = sa_gt_box_of_fg_points
            targets_dict_center['sa_xyz_coords'] = sa_xyz_coords
            targets_dict_center['sa_gt_box_of_points'] = sa_gt_box_of_points
            targets_dict_center['sa_box_idxs_labels'] = sa_box_idxs_labels

        extra_method = target_cfg.get('ASSIGN_METHOD', None)
        if extra_method is not None and extra_method.NAME == 'extend_gt':
            extend_gt_boxes = box_utils.enlarge_box3d(
                gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=extra_method.EXTRA_WIDTH
            ).view(batch_size, -1, gt_boxes.shape[-1])

            if extra_method.get('ASSIGN_TYPE', 'centers') == 'centers_origin': 
                points = input_dict['centers_origin'].detach()
            else:
                points = input_dict['centers'].detach() #default setting

            targets_dict = self.assign_stack_targets_IASSD(
                points=points, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
                set_ignore_flag=True, use_ball_constraint=False,
                ret_box_labels=True,
                use_ex_gt_assign=True, fg_pc_ignore=extra_method.FG_PC_IGNORE,
            )
            targets_dict_center['center_origin_gt_box_of_fg_points'] = targets_dict['gt_box_of_fg_points']
            targets_dict_center['center_origin_cls_labels'] = targets_dict['point_cls_labels']
            targets_dict_center['center_origin_box_idxs_of_pts'] = targets_dict['box_idxs_labels']
            targets_dict_center['gt_box_of_center_origin'] = targets_dict['gt_box_of_points']

        elif extra_method is not None and extra_method.NAME == 'extend_gt_factor':
            extend_gt_boxes = box_utils.enlarge_box3d_with_factor(
                gt_boxes.view(-1, gt_boxes.shape[-1]), factor=extra_method.EXTRA_FACTOR).view(batch_size, -1, gt_boxes.shape[-1])

            if extra_method.get('ASSIGN_TYPE', 'centers') == 'centers_origin': 
                points = input_dict['centers_origin'].detach()
            else:
                points = input_dict['centers'].detach()

            targets_dict = self.assign_stack_targets_IASSD(
                points=points, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
                set_ignore_flag=True, use_ball_constraint=False,
                ret_box_labels=False,
                use_ex_gt_assign=True, fg_pc_ignore=extra_method.FG_PC_IGNORE,
            )
            targets_dict_center['center_origin_gt_box_of_fg_points'] = targets_dict['gt_box_of_fg_points']
            targets_dict_center['center_origin_cls_labels'] = targets_dict['point_cls_labels']

        elif extra_method is not None and extra_method.NAME == 'extend_gt_for_class':
            extend_gt_boxes = box_utils.enlarge_box3d_for_class(
                gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=extra_method.EXTRA_WIDTH
            ).view(batch_size, -1, gt_boxes.shape[-1])

            if extra_method.get('ASSIGN_TYPE', 'centers') == 'centers_origin': 
                points = input_dict['centers_origin'].detach()
            else:
                points = input_dict['centers'].detach()

            targets_dict = self.assign_stack_targets_IASSD(
                points=points, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
                set_ignore_flag=True, use_ball_constraint=False,
                ret_box_labels=False,
                use_ex_gt_assign=True, fg_pc_ignore=extra_method.FG_PC_IGNORE,
            )
            targets_dict_center['center_origin_gt_box_of_fg_points'] = targets_dict['gt_box_of_fg_points']
            targets_dict_center['center_origin_cls_labels'] = targets_dict['point_cls_labels']            

        elif extra_method is not None and extra_method.NAME == 'extend_query':
            extend_gt_boxes = None
            if extra_method.get('ASSIGN_TYPE', 'centers') == 'centers_origin': 
                points = input_dict['centers_origin'].detach()
            elif extra_method.get('ASSIGN_TYPE', 'centers') == 'centers': 
                points = input_dict['centers'].detach()

            targets_dict = self.assign_stack_targets_IASSD(
                points=points, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
                set_ignore_flag=True, use_ball_constraint=False,
                ret_box_labels=False, 
                use_query_assign=True, central_radii=extra_method.RADII, fg_pc_ignore=extra_method.FG_PC_IGNORE,
            )
            targets_dict_center['center_origin_gt_box_of_fg_points'] = targets_dict['gt_box_of_fg_points']
            targets_dict_center['center_origin_cls_labels'] = targets_dict['point_cls_labels']
        
        return targets_dict_center

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict

        # vote loss
        if self.model_cfg.TARGET_CONFIG.get('ASSIGN_METHOD') is not None and \
            self.model_cfg.TARGET_CONFIG.ASSIGN_METHOD.get('ASSIGN_TYPE')== 'centers_origin':
            if self.model_cfg.LOSS_CONFIG.get('LOSS_VOTE_TYPE', 'none') == 'ver1':
                center_loss_reg, tb_dict_3 = self.get_contextual_vote_loss_ver1()
            elif self.model_cfg.LOSS_CONFIG.get('LOSS_VOTE_TYPE', 'none') == 'ver2':
                center_loss_reg, tb_dict_3 = self.get_contextual_vote_loss_ver2()   
             
            else: # 'none'
                center_loss_reg, tb_dict_3 = self.get_contextual_vote_loss()
        else:
            center_loss_reg, tb_dict_3 = self.get_vote_loss_loss() # center assign
        tb_dict.update(tb_dict_3)

        # semantic loss in SA layers
        if self.model_cfg.LOSS_CONFIG.get('LOSS_INS', None) is not None:  # True
            assert ('sa_ins_preds' in self.forward_ret_dict) and ('sa_ins_labels' in self.forward_ret_dict)
            sa_loss_cls, tb_dict_0 = self.get_sa_ins_layer_loss()
       

       
            tb_dict.update(tb_dict_0)
        else:
            sa_loss_cls = 0

        # cls loss  
        center_loss_cls, tb_dict_4 = self.get_center_cls_layer_loss()
        tb_dict.update(tb_dict_4)

        # reg loss
        if self.model_cfg.TARGET_CONFIG.BOX_CODER == 'PointResidualCoder':
            center_loss_box, tb_dict_5 = self.get_box_layer_loss()
        else:
            center_loss_box, tb_dict_5 = self.get_center_box_binori_layer_loss()
        tb_dict.update(tb_dict_5)    
        
        # corner loss
        if self.model_cfg.LOSS_CONFIG.get('CORNER_LOSS_REGULARIZATION', False):
            corner_loss, tb_dict_6 = self.get_corner_layer_loss()
            tb_dict.update(tb_dict_6)

        # iou loss
        iou3d_loss = 0
        if self.model_cfg.LOSS_CONFIG.get('IOU3D_REGULARIZATION', False):
            iou3d_loss, tb_dict_7 = self.get_iou3d_layer_loss()          
            tb_dict.update(tb_dict_7)
        
        point_loss = center_loss_reg + center_loss_cls + center_loss_box + corner_loss + sa_loss_cls + iou3d_loss             
        return point_loss, tb_dict



    def get_contextual_vote_loss(self, tb_dict=None):    
        pos_mask = self.forward_ret_dict['center_origin_cls_labels'] > 0   # [0,1,2,3]
        center_origin_loss_box = []
        for i in self.forward_ret_dict['center_origin_cls_labels'].unique():
            if i <= 0: continue
            simple_pos_mask = self.forward_ret_dict['center_origin_cls_labels'] == i  
           
            center_box_labels = self.forward_ret_dict['center_origin_gt_box_of_fg_points'][:, 0:3][(pos_mask & simple_pos_mask)[pos_mask==1]]
      
            centers_origin = self.forward_ret_dict['centers_origin']
        
            ctr_offsets = self.forward_ret_dict['ctr_offsets']
            
            centers_pred = centers_origin + ctr_offsets 
            centers_pred = centers_pred[simple_pos_mask][:, 1:4] 
            simple_center_origin_loss_box = F.smooth_l1_loss(centers_pred, center_box_labels) 
            center_origin_loss_box.append(simple_center_origin_loss_box.unsqueeze(-1))
            
        center_origin_loss_box = torch.cat(center_origin_loss_box, dim=-1).mean()
        center_origin_loss_box = center_origin_loss_box * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS.get('vote_weight')
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'center_origin_loss_reg': center_origin_loss_box.item()})
        return center_origin_loss_box, tb_dict


    def get_contextual_vote_loss_ver1(self, tb_dict=None):  
        box_idxs_of_pts = self.forward_ret_dict['center_origin_box_idxs_of_pts']
        center_box_labels = self.forward_ret_dict['gt_box_of_center_origin']
        centers_origin = self.forward_ret_dict['centers_origin']
        ctr_offsets = self.forward_ret_dict['ctr_offsets']
        centers_pred = centers_origin[:, 1:] + ctr_offsets[:, 1:]
        centers_pred = torch.cat([centers_origin[:, :1], centers_pred], dim=-1)
        batch_idx = self.forward_ret_dict['centers'][:,0]
        ins_num, ins_vote_loss = [],[]
        for cur_id in batch_idx.unique():
            batch_mask = (batch_idx == cur_id)
            for ins_idx in box_idxs_of_pts[batch_mask].unique():
                if ins_idx < 0:
                    continue
                ins_mask = (box_idxs_of_pts[batch_mask] == ins_idx)
                ins_num.append(ins_mask.sum().long().unsqueeze(-1))
                ins_vote_loss.append(F.smooth_l1_loss(centers_pred[batch_mask][ins_mask, 1:4], center_box_labels[batch_mask][ins_mask, 0:3], reduction='sum').unsqueeze(-1))                
        ins_num = torch.cat(ins_num, dim=-1).float()
        ins_vote_loss = torch.cat(ins_vote_loss, dim=-1)
        ins_vote_loss = ins_vote_loss / ins_num.float().clamp(min=1.0)
        vote_loss = ins_vote_loss.mean()
        vote_loss_ver1 = vote_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['vote_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'vote_loss_ver1': vote_loss_ver1.item()})
        return vote_loss_ver1, tb_dict
  

    def get_contextual_vote_loss_ver2(self, tb_dict=None):  
        box_idxs_of_pts = self.forward_ret_dict['center_origin_box_idxs_of_pts'] 
        center_box_labels = self.forward_ret_dict['gt_box_of_center_origin']
        centers_origin = self.forward_ret_dict['centers_origin'] 
        ctr_offsets = self.forward_ret_dict['ctr_offsets']       
        centers_pred = centers_origin[:, 1:] + ctr_offsets[:, 1:]
        centers_pred = torch.cat([centers_origin[:, :1], centers_pred], dim=-1)     
        batch_idx = self.forward_ret_dict['centers'][:,0]
        ins_num, ins_vote_loss, ins_mean_vote_loss = [],[],[]
    
        for cur_id in batch_idx.unique(): 
            batch_mask = (batch_idx == cur_id)
            for ins_idx in box_idxs_of_pts[batch_mask].unique():
                if ins_idx < 0:
                    continue
                ins_mask = (box_idxs_of_pts[batch_mask] == ins_idx) # box_idxs_of_pts[batch_mask][ins_mask] 
                ins_num.append(ins_mask.sum().unsqueeze(-1))
               
                ins_vote_loss.append(F.smooth_l1_loss(centers_pred[batch_mask][ins_mask, 1:4], center_box_labels[batch_mask][ins_mask, 0:3], reduction='sum').unsqueeze(-1))                     
                # 
                ins_mean_vote_loss.append(F.smooth_l1_loss(centers_pred[batch_mask][ins_mask, 1:4], centers_pred[batch_mask][ins_mask, 1:4].mean(dim=0).repeat(centers_pred[batch_mask][ins_mask, 1:4].shape[0],1), reduction='sum').unsqueeze(-1))                
        if len(ins_num)==0:
            ins_num = torch.zeros((1)).float() 
            ins_vote_loss = torch.zeros((1)).float()  
            ins_mean_vote_loss = torch.zeros((1)).float()   
        else:
            ins_num = torch.cat(ins_num, dim=-1).float() 
            ins_vote_loss = torch.cat(ins_vote_loss, dim=-1) 
            ins_mean_vote_loss = torch.cat(ins_mean_vote_loss, dim=-1)  
        vote_loss = ins_vote_loss + ins_mean_vote_loss * 0.5
        vote_loss = vote_loss / ins_num.clamp(min=1.0) 
        vote_loss = vote_loss.mean()
        vote_loss_ver2 = vote_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['vote_weight']
        
        
        
        
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'vote_loss_ver2': vote_loss_ver2.item()})
        return vote_loss_ver2, tb_dict


    def get_vote_loss_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['center_cls_labels'] > 0
        center_box_labels = self.forward_ret_dict['center_gt_box_of_fg_points'][:, 0:3]
        centers_origin = self.forward_ret_dict['centers_origin']
        ctr_offsets = self.forward_ret_dict['ctr_offsets']
        centers_pred = centers_origin + ctr_offsets
        centers_pred = centers_pred[pos_mask][:, 1:4]

        vote_loss = F.smooth_l1_loss(centers_pred, center_box_labels, reduction='mean')
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'vote_loss': vote_loss.item()})
        return vote_loss, tb_dict


    def get_center_cls_layer_loss(self, tb_dict=None):
        point_cls_labels = self.forward_ret_dict['center_cls_labels'].view(-1)
        point_cls_preds = self.forward_ret_dict['center_cls_preds'].view(-1, self.num_class)
        positives = (point_cls_labels > 0)
        negative_cls_weights = (point_cls_labels == 0) * 1.0

        cls_weights = (1.0 *negative_cls_weights + 1.0 * positives).float()
        pos_normalizer = positives.sum(dim=0).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
        one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:]
        
        if self.model_cfg.LOSS_CONFIG.CENTERNESS_REGULARIZATION:
            centerness_mask = self.generate_center_ness_mask()
            one_hot_targets = one_hot_targets * centerness_mask.unsqueeze(-1).repeat(1, one_hot_targets.shape[1])

        point_loss_cls = self.cls_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights).mean(dim=-1).sum()
        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_cls = point_loss_cls * loss_weights_dict['point_cls_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({
            'center_loss_cls': point_loss_cls.item(),
            'center_pos_num': pos_normalizer.item()
        })
        return point_loss_cls, tb_dict



    def get_sa_ins_layer_loss(self, tb_dict=None):
        sa_ins_labels = self.forward_ret_dict['sa_ins_labels'] # list[5]，每个不管是不是 ctr 都产生了 每个标签值 长度为[B*N]
        sa_ins_preds = self.forward_ret_dict['sa_ins_preds']  # 产生 MAsk  第一个是batch的索引  list[5] 不是ctr的地方都是空的，为ctr的部分  [B,N,4]
     
        
        # sa_centerness_mask,sa_centerness_topk  = self.gauss_fun_once_topk() # 这是加上了 CD Loss的
        sa_centerness_mask,sa_centerness_topk  = self.gauss_fun_once_topk_GT_add_same_size() # 这是加上了 CD Loss的

        sa_ins_loss, ignore = 0, 0
        
        
        
        for i in range(len(sa_ins_labels)): # valid when i =1, 2  对每一层
            if len(sa_ins_preds[i]) != 0:  # 其实整个for循环还是计算 ctr层的损失，continue直接就把后面的都跳过了
                try:
                    point_cls_preds = sa_ins_preds[i][...,1:].view(-1, self.num_class)   # 将batch index 去掉
                except:
                    point_cls_preds = sa_ins_preds[i][...,1:].view(-1, 1)
            else:
                ignore += 1
                continue
            point_cls_labels = sa_ins_labels[i].view(-1)  #
            positives = (point_cls_labels > 0)
            negative_cls_weights = (point_cls_labels == 0) * 1.0  # 操作就是将 True False变为 1，0
            cls_weights = (negative_cls_weights + 1.0 * positives).float() # --> 全是1
            pos_normalizer = positives.sum(dim=0).float() # 正样本的个数
            cls_weights /= torch.clamp(pos_normalizer, min=1.0)  # 归一化
            # [N*B,4]:4为三个类别加个前景点 好形成 one-hot编码
            one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
            one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)  # 根据中间的0，1，2，3 将1散落在对应的索引
            one_hot_targets = one_hot_targets[..., 1:]
            # 这边就是判断是否需要 mask的
            if ('ctr' in self.model_cfg.LOSS_CONFIG.SAMPLE_METHOD_LIST[i+1][0]):
                centerness_mask = sa_centerness_mask[i] # 每层每个前景点的mask值，背景点为mask值0
                one_hot_targets = one_hot_targets * centerness_mask.unsqueeze(-1).repeat(1, one_hot_targets.shape[1])  # [B*N,3] *([B*N]-->[B*N,1]--->repeat[B*N,3]
                # 应该是one_hot 编码，其实其他地方本来就是0，所以对应位置的值还是那么多
                # 根据label中的索引[3]: 0，1，2，3
                # 将依次取得三个值
           
            # point_cls_preds:B*N,class   one_hot_targets:里面行顶多一个非0值
            point_loss_ins = self.ins_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights).mean(dim=-1).sum()        
            loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
            point_loss_ins = point_loss_ins * loss_weights_dict.get('ins_aware_weight',[1]*len(sa_ins_labels))[i]

            sa_ins_loss += point_loss_ins
            if tb_dict is None:
                tb_dict = {}
            tb_dict.update({
                'sa%s_loss_ins' % str(i): point_loss_ins.item(),
                'sa%s_pos_num' % str(i): pos_normalizer.item()
            })

        sa_ins_loss = sa_ins_loss / (len(sa_ins_labels) - ignore)
        
        
     
        
        tb_dict.update({
                'sa_loss_ins': sa_ins_loss.item(),
                'CD_loss' : sa_centerness_topk.item()
            })
        
        sa_ins_loss = sa_ins_loss #+0.8*sa_centerness_topk
        
        
        tb_dict.update({
                'sa_loss_ins_all': sa_ins_loss.item(),
            })
        return sa_ins_loss, tb_dict
    
    def get_sa_ins_layer_loss_imbalance(self, tb_dict=None):
        sa_ins_labels = self.forward_ret_dict['sa_ins_labels'] # list[5]，每个不管是不是 ctr 都产生了 每个标签值 长度为[B*N]
        sa_ins_preds = self.forward_ret_dict['sa_ins_preds']  # 产生 MAsk  第一个是batch的索引  list[5] 不是ctr的地方都是空的，为ctr的部分  [B,N,4]
        sa_centerness_mask = self.generate_sa_center_ness_mask()  
        # sa_centerness_mask = self.gauss_fun()
        sa_ins_loss, ignore = 0, 0
        for i in range(len(sa_ins_labels)): # valid when i =1, 2  对每一层
            if len(sa_ins_preds[i]) != 0:  # 其实整个for循环还是计算 ctr层的损失，continue直接就把后面的都跳过了
                try:
                    point_cls_preds = sa_ins_preds[i][...,1:].view(-1, self.num_class)   # 将batch index 去掉
                except:
                    point_cls_preds = sa_ins_preds[i][...,1:].view(-1, 1)
            else:
                ignore += 1
                continue
            point_cls_labels = sa_ins_labels[i].view(-1)
            positives = (point_cls_labels > 0)
            
            # weights = torch.where(point_cls_labels ==1 , torch.tensor(2.0), x)
            
            
            negative_cls_weights = (point_cls_labels == 0) * 1.0  # 操作就是将 True False变为 1，0
            cls_weights = (negative_cls_weights + 1.0 * positives).float() # --> 全是1
            pos_normalizer = positives.sum(dim=0).float() # 正样本的个数
            cls_weights /= torch.clamp(pos_normalizer, min=1.0)  # 归一化
            # [N*B,4]:4为三个类别加个前景点 好形成 one-hot编码
            one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
            one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)  # 根据中间的0，1，2，3 将1散落在对应的索引
            one_hot_targets = one_hot_targets[..., 1:]
            # 这边就是判断是否需要 mask的
            if ('ctr' in self.model_cfg.LOSS_CONFIG.SAMPLE_METHOD_LIST[i+1][0]):
                centerness_mask = sa_centerness_mask[i] # 每层每个前景点的mask值，背景点为mask值0
                one_hot_targets = one_hot_targets * centerness_mask.unsqueeze(-1).repeat(1, one_hot_targets.shape[1])  # [B*N,3] *([B*N]-->[B*N,1]--->repeat[B*N,3]
                # 应该是one_hot 编码，其实其他地方本来就是0，所以对应位置的值还是那么多
                # 根据label中的索引[3]: 0，1，2，3
          

            point_loss_ins = self.ins_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights).mean(dim=-1).sum()        
            loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
            point_loss_ins = point_loss_ins * loss_weights_dict.get('ins_aware_weight',[1]*len(sa_ins_labels))[i]

            sa_ins_loss += point_loss_ins
            if tb_dict is None:
                tb_dict = {}
            tb_dict.update({
                'sa%s_loss_ins' % str(i): point_loss_ins.item(),
                'sa%s_pos_num' % str(i): pos_normalizer.item()
            })

        sa_ins_loss = sa_ins_loss / (len(sa_ins_labels) - ignore)
        tb_dict.update({
                'sa_loss_ins': sa_ins_loss.item(),
            })
        return sa_ins_loss, tb_dict



    def generate_center_ness_mask(self):
        pos_mask = self.forward_ret_dict['center_cls_labels'] > 0
        gt_boxes = self.forward_ret_dict['center_gt_box_of_fg_points']
        centers = self.forward_ret_dict['centers'][:,1:]
        centers = centers[pos_mask].clone().detach()
        offset_xyz = centers[:, 0:3] - gt_boxes[:, 0:3]

        offset_xyz_canical = common_utils.rotate_points_along_z(offset_xyz.unsqueeze(dim=1), -gt_boxes[:, 6]).squeeze(dim=1)

        template = gt_boxes.new_tensor(([1, 1, 1], [-1, -1, -1])) / 2
        margin = gt_boxes[:, None, 3:6].repeat(1, 2, 1) * template[None, :, :]
        distance = margin - offset_xyz_canical[:, None, :].repeat(1, 2, 1)
        distance[:, 1, :] = -1 * distance[:, 1, :]
        distance_min = torch.where(distance[:, 0, :] < distance[:, 1, :], distance[:, 0, :], distance[:, 1, :])
        distance_max = torch.where(distance[:, 0, :] > distance[:, 1, :], distance[:, 0, :], distance[:, 1, :])

        centerness = distance_min / distance_max
        centerness = centerness[:, 0] * centerness[:, 1] * centerness[:, 2]
        centerness = torch.clamp(centerness, min=1e-6)
        centerness = torch.pow(centerness, 1/3)

        centerness_mask = pos_mask.new_zeros(pos_mask.shape).float()
        centerness_mask[pos_mask] = centerness
        return centerness_mask
    
 
    
    def gauss_fun_kitti(self):  
        sa_pos_mask = self.forward_ret_dict['sa_ins_labels'] # list[[B*N],...,] 5个 点云目标检测任务中的实例标签  sa应该是sample的意思应该   list[5]，每个不管是不是 ctr 都产生了 每个标签值 长度为[B*N] -1,0,1,2,3
        sa_gt_boxes = self.forward_ret_dict['sa_gt_box_of_fg_points'] # # list[[B*N，8],...,] 5个 能包含了每个前景点云目标的真实边界框信息，包括边界框的位置、尺寸和方向等  八位：最后一位是类别
        sa_xyz_coords = self.forward_ret_dict['sa_xyz_coords'] # list[[B,N,4],,,] 5个 第一位是batch的索引
        sa_centerness_mask = []
        for i in range(len(sa_pos_mask)): # 对每一层
            pos_mask = sa_pos_mask[i] > 0 # 大于0的点的mask
            gt_boxes = sa_gt_boxes[i]   # 真实的点对应框
            xyz_coords = sa_xyz_coords[i].view(-1,sa_xyz_coords[i].shape[-1])[:,1:] # [B*N,3]
            xyz_coords = xyz_coords[pos_mask].clone().detach() # 得到分数大于0的点
            offset_xyz = xyz_coords[:, 0:3] - gt_boxes[:, 0:3] # 得到和GT框中心的偏移量，也考验理解为转到局部坐标系  # （openpcdet中，x向前，y向左，z向上，x到y逆时针为正）
            offset_xyz_canical = common_utils.rotate_points_along_z(offset_xyz.unsqueeze(dim=1), -gt_boxes[:, 6]).squeeze(dim=1)#  # 上面完成平移操作后，下面完成旋转操作，使x方向朝向车头方向，y垂直于x，z向上 

       
            w_gt = gt_boxes[:, 3]  
            l_gt = gt_boxes[:, 4]
            h_gt = gt_boxes[:, 5]
      
            _COVARIANCE_1 = 4/(w_gt ** 2 + l_gt ** 2) 
            _COVARIANCE_2 = 4/(w_gt ** 2 + h_gt ** 2)
            _COVARIANCE_3 = 4/(h_gt ** 2 + l_gt ** 2)
            
            
            # 下面这是once的  # 'Car:5', 'Bus:5', 'Truck:5', 'Pedestrian:5', 'Cyclist:5'
            # _COVARIANCE_1 = torch.where(gt_boxes[:,-1]==1,_COVARIANCE_1*4,_COVARIANCE_1)
            # _COVARIANCE_1 = torch.where(gt_boxes[:,-1]==2,_COVARIANCE_1*6,_COVARIANCE_1)
            # _COVARIANCE_1 = torch.where(gt_boxes[:,-1]==3,_COVARIANCE_1*5,_COVARIANCE_1)
            
            # _COVARIANCE_2 = torch.where(gt_boxes[:,-1]==1,_COVARIANCE_2*4,_COVARIANCE_2)
            # _COVARIANCE_2 = torch.where(gt_boxes[:,-1]==2,_COVARIANCE_2*6,_COVARIANCE_2)
            # _COVARIANCE_2 = torch.where(gt_boxes[:,-1]==3,_COVARIANCE_2*5,_COVARIANCE_2)
            
            # _COVARIANCE_3 = torch.where(gt_boxes[:,-1]==1,_COVARIANCE_3*4,_COVARIANCE_3)
            # _COVARIANCE_3 = torch.where(gt_boxes[:,-1]==2,_COVARIANCE_3*6,_COVARIANCE_3)
            # _COVARIANCE_3 = torch.where(gt_boxes[:,-1]==3,_COVARIANCE_3*5,_COVARIANCE_3)
            
            # # 这是 kitti ：'Car:5', 'Pedestrian:5', 'Cyclist:5'
            _COVARIANCE_1 = torch.where(gt_boxes[:,-1]==1,_COVARIANCE_1*3,_COVARIANCE_1)
            _COVARIANCE_2 = torch.where(gt_boxes[:,-1]==1,_COVARIANCE_2*3,_COVARIANCE_2)
            _COVARIANCE_3 = torch.where(gt_boxes[:,-1]==1,_COVARIANCE_3*3,_COVARIANCE_3)

            
            offset_xyz_canical[:,0] = offset_xyz_canical[:,0]*_COVARIANCE_1
            offset_xyz_canical[:,1] = offset_xyz_canical[:,1]*_COVARIANCE_2
            offset_xyz_canical[:,2] = offset_xyz_canical[:,2]*_COVARIANCE_3
            
            # offset_xyz_canical:N,3
            # 
            
            # # torch.matmul   这不就是距离吗
            value_matric = torch.mm(offset_xyz_canical,offset_xyz_canical.t())  # 一个2D张量，形状为(m, p)，表示mat1与mat2的矩阵乘法结果。
            diag_value = torch.diag(value_matric) # 这应该相当于就是 l2距离吧，算平方但是没有开根号
            gt_hm = torch.exp(-0.5 * diag_value) # [B,N]
            
            centerness_mask = pos_mask.new_zeros(pos_mask.shape).float() # [B,N]
            centerness_mask[pos_mask] = gt_hm # 对前景点赋 这个距离值

            sa_centerness_mask.append(centerness_mask) # 加到列表中
            
            

        return sa_centerness_mask
    
 
 
    
  
    def gauss_fun_once_topk_GT_add_same_size(self):  
        sa_pos_mask = self.forward_ret_dict['sa_ins_labels'] # list[[B*N],...,] 5个 点云目标检测任务中的实例标签  sa应该是sample的意思应该   list[5]，每个不管是不是 ctr 都产生了 每个标签值 长度为[B*N] -1,0,1,2,3
        sa_gt_boxes = self.forward_ret_dict['sa_gt_box_of_fg_points'] # # list[[B*N，8],...,] 5个 能包含了每个前景点云目标的真实边界框信息，包括边界框的位置、尺寸和方向等  八位：最后一位是类别
        sa_xyz_coords = self.forward_ret_dict['sa_xyz_coords'] # list[[B,N,4],,,] 5个 第一位是batch的索引
        sa_centerness_mask = []
        
        sa_centerness_topk = []
        
        for i in range(len(sa_pos_mask)): # 对每一层
            pos_mask = sa_pos_mask[i] > 0 # 大于0的点的mask
            gt_boxes = sa_gt_boxes[i]   # 真实的点对应框
            xyz_coords = sa_xyz_coords[i].view(-1,sa_xyz_coords[i].shape[-1])[:,1:] # [B*N,3]
            xyz_coords = xyz_coords[pos_mask].clone().detach() # 得到分数大于0的点
            offset_xyz = xyz_coords[:, 0:3] - gt_boxes[:, 0:3] # 得到和GT框中心的偏移量，也考验理解为转到局部坐标系  # （openpcdet中，x向前，y向左，z向上，x到y逆时针为正）
            offset_xyz_canical = common_utils.rotate_points_along_z(offset_xyz.unsqueeze(dim=1), -gt_boxes[:, 6]).squeeze(dim=1)#  # 上面完成平移操作后，下面完成旋转操作，使x方向朝向车头方向，y垂直于x，z向上 

            
            w_gt = gt_boxes[:, 3] 
            l_gt = gt_boxes[:, 4]
            h_gt = gt_boxes[:, 5]
            _COVARIANCE_1 = 4/(w_gt ** 2 + l_gt ** 2)  
            _COVARIANCE_2 = 4/(w_gt ** 2 + h_gt ** 2)
            _COVARIANCE_3 = 4/(h_gt ** 2 + l_gt ** 2)
            
            
      
            """
            _COVARIANCE_1 = 4/(w_gt ** 2 + l_gt ** 2)  
            _COVARIANCE_2 = 4/(w_gt ** 2 + l_gt ** 2)
            _COVARIANCE_3 = 4/(h_gt ** 2 )
            """
            
            # 下面这是once的  # 'Car:5', 'Bus:5', 'Truck:5', 'Pedestrian:5', 'Cyclist:5'
            _COVARIANCE_1 = torch.where(gt_boxes[:,-1]==1,_COVARIANCE_1*4,_COVARIANCE_1)
            _COVARIANCE_1 = torch.where(gt_boxes[:,-1]==2,_COVARIANCE_1*6,_COVARIANCE_1)
            _COVARIANCE_1 = torch.where(gt_boxes[:,-1]==3,_COVARIANCE_1*5,_COVARIANCE_1)
            
            _COVARIANCE_2 = torch.where(gt_boxes[:,-1]==1,_COVARIANCE_2*4,_COVARIANCE_2)
            _COVARIANCE_2 = torch.where(gt_boxes[:,-1]==2,_COVARIANCE_2*6,_COVARIANCE_2)
            _COVARIANCE_2 = torch.where(gt_boxes[:,-1]==3,_COVARIANCE_2*5,_COVARIANCE_2)
            
            _COVARIANCE_3 = torch.where(gt_boxes[:,-1]==1,_COVARIANCE_3*4,_COVARIANCE_3)
            _COVARIANCE_3 = torch.where(gt_boxes[:,-1]==2,_COVARIANCE_3*6,_COVARIANCE_3)
            _COVARIANCE_3 = torch.where(gt_boxes[:,-1]==3,_COVARIANCE_3*5,_COVARIANCE_3)            
            
            offset_xyz_canical[:,0] = offset_xyz_canical[:,0]*_COVARIANCE_1
            offset_xyz_canical[:,1] = offset_xyz_canical[:,1]*_COVARIANCE_2
            offset_xyz_canical[:,2] = offset_xyz_canical[:,2]*_COVARIANCE_3
            
      
            
          
            value_matric = torch.mm(offset_xyz_canical,offset_xyz_canical.t())  # 一个2D张量，形状为(m, p)，表示mat1与mat2的矩阵乘法结果。
            diag_value = torch.diag(value_matric) # 这应该相当于就是 l2距离吧，算平方但是没有开根号
            gt_hm = torch.exp(-0.5 * diag_value) # [B,N]
            
            centerness_mask = pos_mask.new_zeros(pos_mask.shape).float() # [B*N]
            centerness_mask[pos_mask] = gt_hm # 对前景点赋 这个距离值

            sa_centerness_mask.append(centerness_mask) # 加到列表中
            
            
            
            if i+1 < len(sa_pos_mask): # 前四个，获得下一层的大小
                
                batch_size = sa_xyz_coords[i].shape[0]
                batch_idx_start = 0
                batch_idx_end = centerness_mask.shape[0] // batch_size
                
                sample_number = sa_pos_mask[i+1].shape[0] // batch_size
                
                sa_centerness_topk_batch = []
                
                pos_fg_number_start=0
                
                
                for batch_i in range(batch_size):
                    centerness_mask_batch = centerness_mask[batch_idx_start:batch_idx_end] # 第i 个bacth_size 中的n点
                    batch_idx_start = batch_idx_end
                    batch_idx_end  = batch_idx_end +batch_idx_end 
                    # 取得下一层中的点
                    score_picked, sample_idx = torch.topk(centerness_mask_batch, sample_number, dim=-1)    # 获得4096个点的索引  
                    
     
                    # 这两步是为了只保留topk中选出的 前景点，相当于里面的前景点都只有这么多
                    # 在 topk 中找到大于 0 的值的索引
                    nonzero_indices = torch.nonzero(score_picked > 0, as_tuple=False)
                    # 得到 topk 中大于 0 的值对应的索引
                    sample_idx = sample_idx[nonzero_indices].squeeze()           
                  
                    
                    
                    ner_xyz_coord = sa_xyz_coords[i][batch_i,:,1:]  # n,3
                    
                    new_fg_xyz = ner_xyz_coord[sample_idx].clone().detach()
                    
                    
                    pos_fg_number = sum(centerness_mask_batch > 0)  # 当前batch 中前景点的个数 tensor:int
                    if len(sample_idx) <  sample_number:
                        pos_mask_add = centerness_mask_batch <= 0 # 大于0的点的mask [N]
                        gt_boxes_bk = sa_gt_boxes[i][pos_fg_number_start:pos_fg_number_start+pos_fg_number][:, 0:3]   # 真实的点对应框 [N,3]
                        pos_fg_number_start = pos_fg_number_start+pos_fg_number
                   
                           
                        ner_xyz_coord = sa_xyz_coords[i][batch_i,:,1:].clone().detach() # n,3 # .unsqueeze(0).transpose(1, 2).contiguous()  # b,3,n
                        # 找到该batch的其他的背景点
                        xyz_bg = ner_xyz_coord[pos_mask_add] # n,3  14391
                        # 计算背景点的距离
                        
                        # Expand dimensions to compute squared distances
                        points1_expanded = xyz_bg.unsqueeze(1) # n,1,3
                        points2_expanded = gt_boxes_bk.unsqueeze(0)  # 1,m,3
                        
                        # Compute squared L2 distance
                        distances_squared = torch.sum((points1_expanded - points2_expanded) ** 2, dim=-1)
                        
                        # 得到每一行的最小值 [14391]
                        min_values, _  = torch.min(distances_squared, dim=1, keepdim=False) # points1_expanded 个数 
                    
                        _, indices = torch.topk(min_values, k=sample_number-len(sample_idx), dim=-1, largest=False) 
                        remind_new_xyz_bg = xyz_bg[indices ]

                        new_fg_xyz = torch.cat([new_fg_xyz,remind_new_xyz_bg])
                     
                    sa_centerness_topk_batch.append(new_fg_xyz)
                
                sa_centerness_topk.append(sa_centerness_topk_batch)                  
                
        
        cdloss_xyz_loss = []
        for i in range(4):     
            # 总的
            if  ('ctr' in self.model_cfg.LOSS_CONFIG.SAMPLE_METHOD_LIST[i][0]):  # 248
                
                batch_size = sa_xyz_coords[i].shape[0]
                batch_idx_start = 0
                batch_idx_end = centerness_mask.shape[0] // batch_size
                sample_number = sa_pos_mask[i+1].shape[0] // batch_size
                
                
                batch_cd_loss=[]
                for batch_i in range(batch_size):
                    # use CD distance
                    pred_xyz_coords = sa_xyz_coords[i][batch_i,:,1:].unsqueeze(0)  # [1,N,3]
                    sa_centerness_topk_xyz_coords = sa_centerness_topk[i-1][batch_i].unsqueeze(0) 
                    
              
                    cdloss_xyz_batch = cd_loss.cd_loss_L1(pred_xyz_coords,sa_centerness_topk_xyz_coords)
                    batch_cd_loss.append(cdloss_xyz_batch )
                    
                cdloss_xyz= sum(batch_cd_loss)/len(batch_cd_loss) 
                cdloss_xyz_loss.append(cdloss_xyz)
            
        return sa_centerness_mask,sum(cdloss_xyz_loss)/len(cdloss_xyz_loss)    # 每一层每个点的分数
   
    

    

    def gauss_fun_once_topk(self):  
        sa_pos_mask = self.forward_ret_dict['sa_ins_labels']
        sa_gt_boxes = self.forward_ret_dict['sa_gt_box_of_fg_points'] 
        sa_xyz_coords = self.forward_ret_dict['sa_xyz_coords'] 
        sa_centerness_mask = []
        
        sa_centerness_topk = []
        
        for i in range(len(sa_pos_mask)): # 对每一层
            pos_mask = sa_pos_mask[i] > 0 # 大于0的点的mask
            gt_boxes = sa_gt_boxes[i]   # 真实的点对应框
            xyz_coords = sa_xyz_coords[i].view(-1,sa_xyz_coords[i].shape[-1])[:,1:] # [B*N,3]
            xyz_coords = xyz_coords[pos_mask].clone().detach() # 得到分数大于0的点
            offset_xyz = xyz_coords[:, 0:3] - gt_boxes[:, 0:3] # 得到和GT框中心的偏移量，也考验理解为转到局部坐标系  # （openpcdet中，x向前，y向左，z向上，x到y逆时针为正）
            offset_xyz_canical = common_utils.rotate_points_along_z(offset_xyz.unsqueeze(dim=1), -gt_boxes[:, 6]).squeeze(dim=1)#  # 上面完成平移操作后，下面完成旋转操作，使x方向朝向车头方向，y垂直于x，z向上 

         
            w_gt = gt_boxes[:, 3]  
            l_gt = gt_boxes[:, 4]
            h_gt = gt_boxes[:, 5]
         
            _COVARIANCE_1 = 4/(w_gt ** 2 + l_gt ** 2)  #
            _COVARIANCE_2 = 4/(w_gt ** 2 + h_gt ** 2)
            _COVARIANCE_3 = 4/(h_gt ** 2 + l_gt ** 2)
        
          
            _COVARIANCE_1 = torch.where(gt_boxes[:,-1]==1,_COVARIANCE_1*4,_COVARIANCE_1)
            _COVARIANCE_1 = torch.where(gt_boxes[:,-1]==2,_COVARIANCE_1*6,_COVARIANCE_1)
            _COVARIANCE_1 = torch.where(gt_boxes[:,-1]==3,_COVARIANCE_1*5,_COVARIANCE_1)
            
            _COVARIANCE_2 = torch.where(gt_boxes[:,-1]==1,_COVARIANCE_2*4,_COVARIANCE_2)
            _COVARIANCE_2 = torch.where(gt_boxes[:,-1]==2,_COVARIANCE_2*6,_COVARIANCE_2)
            _COVARIANCE_2 = torch.where(gt_boxes[:,-1]==3,_COVARIANCE_2*5,_COVARIANCE_2)
            
            _COVARIANCE_3 = torch.where(gt_boxes[:,-1]==1,_COVARIANCE_3*4,_COVARIANCE_3)
            _COVARIANCE_3 = torch.where(gt_boxes[:,-1]==2,_COVARIANCE_3*6,_COVARIANCE_3)
            _COVARIANCE_3 = torch.where(gt_boxes[:,-1]==3,_COVARIANCE_3*5,_COVARIANCE_3)     
            
        
            offset_xyz_canical[:,0] = offset_xyz_canical[:,0]*_COVARIANCE_1
            offset_xyz_canical[:,1] = offset_xyz_canical[:,1]*_COVARIANCE_2
            offset_xyz_canical[:,2] = offset_xyz_canical[:,2]*_COVARIANCE_3
            
         
            # # torch.matmul   这不就是距离吗
            value_matric = torch.mm(offset_xyz_canical,offset_xyz_canical.t())  # 一个2D张量，形状为(m, p)，表示mat1与mat2的矩阵乘法结果。
            diag_value = torch.diag(value_matric) # 这应该相当于就是 l2距离吧，算平方但是没有开根号
            gt_hm = torch.exp(-0.5 * diag_value) # [B,N]
            
            centerness_mask = pos_mask.new_zeros(pos_mask.shape).float() # [B,N]
            centerness_mask[pos_mask] = gt_hm # 对前景点赋 这个距离值

            sa_centerness_mask.append(centerness_mask) # 加到列表中
            
            
            
            if i+1 < len(sa_pos_mask): # 前四个，获得下一层的大小
                score_picked, sample_idx = torch.topk(centerness_mask, sa_pos_mask[i+1].shape[0], dim=-1)    # 获得4096个点的索引  
                
                              

                # 这两步是为了只保留topk中选出的 前景点，相当于里面的前景点都只有这么多
                # 在 topk 中找到大于 0 的值的索引
                nonzero_indices = torch.nonzero(score_picked > 0, as_tuple=False)

                # 得到 topk 中大于 0 的值对应的索引
                sample_idx = sample_idx[nonzero_indices].squeeze()
                
                
                sample_idx = sample_idx.int()    
                # sample_idx = sample_idx.to(torch.int64)
                # xyz_coords[i][sample_idx]
                # torch.gather(xyz_coords[i],0,sample_idx)
                ner_xyz_coord = sa_xyz_coords[i].view(-1,sa_xyz_coords[i].shape[-1])[:,1:].unsqueeze(0).transpose(1, 2).contiguous()  # b,3,n
                sample_idx = sample_idx.unsqueeze(0)
                # 4096,3
                new_xyz_coords = pointnet2_utils.gather_operation(ner_xyz_coord, sample_idx).transpose(1, 2).contiguous().squeeze(0)
                sa_centerness_topk.append(new_xyz_coords)
 
                
                
        
        cdloss_xyz_loss = []
        for i in range(4):     
            # 总的
            if  ('ctr' in self.model_cfg.LOSS_CONFIG.SAMPLE_METHOD_LIST[i][0]):  # 248
                # 使用 CD 距离
                pred_xyz_coords = sa_xyz_coords[i].view(-1,sa_xyz_coords[i].shape[-1])[:,1:].unsqueeze(0)
                sa_centerness_topk_xyz_coords = sa_centerness_topk[i-1].unsqueeze(0)
                cdloss_xyz = cd_loss.cd_loss_L1(pred_xyz_coords,sa_centerness_topk_xyz_coords)
                # cdloss_xyz = cd_loss.cd_loss_L2(pred_xyz_coords,sa_centerness_topk_xyz_coords)
                cdloss_xyz_loss.append(cdloss_xyz)
            
        return sa_centerness_mask,sum(cdloss_xyz_loss)/len(cdloss_xyz_loss)    # 每一层每个点的分数
        # return sa_centerness_mask
    
  

    # 计算高斯距离
    def gauss_fun_once(self):  
        sa_pos_mask = self.forward_ret_dict['sa_ins_labels'] # list[[B*N],...,] 5个 点云目标检测任务中的实例标签  sa应该是sample的意思应该   list[5]，每个不管是不是 ctr 都产生了 每个标签值 长度为[B*N] -1,0,1,2,3
        sa_gt_boxes = self.forward_ret_dict['sa_gt_box_of_fg_points'] # # list[[B*N，8],...,] 5个 能包含了每个前景点云目标的真实边界框信息，包括边界框的位置、尺寸和方向等  八位：最后一位是类别
        sa_xyz_coords = self.forward_ret_dict['sa_xyz_coords'] # list[[B,N,4],,,] 5个 第一位是batch的索引
        sa_centerness_mask = []
        for i in range(len(sa_pos_mask)): # 对每一层
            pos_mask = sa_pos_mask[i] > 0 # 大于0的点的mask
            gt_boxes = sa_gt_boxes[i]   # 真实的点对应框
            xyz_coords = sa_xyz_coords[i].view(-1,sa_xyz_coords[i].shape[-1])[:,1:] # [B*N,3]
            xyz_coords = xyz_coords[pos_mask].clone().detach() # 得到分数大于0的点
            offset_xyz = xyz_coords[:, 0:3] - gt_boxes[:, 0:3] # 得到和GT框中心的偏移量，也考验理解为转到局部坐标系  # （openpcdet中，x向前，y向左，z向上，x到y逆时针为正）
            offset_xyz_canical = common_utils.rotate_points_along_z(offset_xyz.unsqueeze(dim=1), -gt_boxes[:, 6]).squeeze(dim=1)#  # 上面完成平移操作后，下面完成旋转操作，使x方向朝向车头方向，y垂直于x，z向上 

            w_gt = gt_boxes[:, 3]  
            l_gt = gt_boxes[:, 4]
            h_gt = gt_boxes[:, 5]
    
            _COVARIANCE_1 = 4/(w_gt ** 2 + l_gt ** 2)  # 这是对角矩阵的逆
            _COVARIANCE_2 = 4/(w_gt ** 2 + h_gt ** 2)
            _COVARIANCE_3 = 4/(h_gt ** 2 + l_gt ** 2)
            
            
            # 参照 pointrcnn的 boxed-encod 来实现 
            """
            _COVARIANCE_1 = 4/(w_gt ** 2 + l_gt ** 2)  # 这是对角矩阵的逆
            _COVARIANCE_2 = 4/(w_gt ** 2 + l_gt ** 2)
            _COVARIANCE_3 = 4/(h_gt ** 2 )
            """
            
            # 下面这是once的  # 'Car:5', 'Bus:5', 'Truck:5', 'Pedestrian:5', 'Cyclist:5'
            _COVARIANCE_1 = torch.where(gt_boxes[:,-1]==1,_COVARIANCE_1*4,_COVARIANCE_1)
            _COVARIANCE_1 = torch.where(gt_boxes[:,-1]==2,_COVARIANCE_1*6,_COVARIANCE_1)
            _COVARIANCE_1 = torch.where(gt_boxes[:,-1]==3,_COVARIANCE_1*5,_COVARIANCE_1)
            
            _COVARIANCE_2 = torch.where(gt_boxes[:,-1]==1,_COVARIANCE_2*4,_COVARIANCE_2)
            _COVARIANCE_2 = torch.where(gt_boxes[:,-1]==2,_COVARIANCE_2*6,_COVARIANCE_2)
            _COVARIANCE_2 = torch.where(gt_boxes[:,-1]==3,_COVARIANCE_2*5,_COVARIANCE_2)
            
            _COVARIANCE_3 = torch.where(gt_boxes[:,-1]==1,_COVARIANCE_3*4,_COVARIANCE_3)
            _COVARIANCE_3 = torch.where(gt_boxes[:,-1]==2,_COVARIANCE_3*6,_COVARIANCE_3)
            _COVARIANCE_3 = torch.where(gt_boxes[:,-1]==3,_COVARIANCE_3*5,_COVARIANCE_3)


            offset_xyz_canical[:,0] = offset_xyz_canical[:,0]*_COVARIANCE_1
            offset_xyz_canical[:,1] = offset_xyz_canical[:,1]*_COVARIANCE_2
            offset_xyz_canical[:,2] = offset_xyz_canical[:,2]*_COVARIANCE_3
            

            value_matric = torch.mm(offset_xyz_canical,offset_xyz_canical.t())  # 一个2D张量，形状为(m, p)，表示mat1与mat2的矩阵乘法结果。
            diag_value = torch.diag(value_matric) # 这应该相当于就是 l2距离吧，算平方但是没有开根号
            gt_hm = torch.exp(-0.5 * diag_value) # [B,N]
            
            centerness_mask = pos_mask.new_zeros(pos_mask.shape).float() # [B,N]
            centerness_mask[pos_mask] = gt_hm # 对前景点赋 这个距离值

            sa_centerness_mask.append(centerness_mask) # 加到列表中
            

        return sa_centerness_mask

    def generate_sa_center_ness_mask(self): # 这些点的标签都是波动产生的，看哪个产生采样点之后得到的标签值
        sa_pos_mask = self.forward_ret_dict['sa_ins_labels'] # 点云目标检测任务中的实例标签  sa应该是sample的意思应该   list[5]，每个不管是不是 ctr 都产生了 每个标签值 长度为[B*N] -1,0,1,2,3
        sa_gt_boxes = self.forward_ret_dict['sa_gt_box_of_fg_points'] # 能包含了每个前景点云目标的真实边界框信息，包括边界框的位置、尺寸和方向等  八位：最后一位是类别
        sa_xyz_coords = self.forward_ret_dict['sa_xyz_coords'] # [B,N,4] 第一位是batch的索引
        sa_centerness_mask = []
        for i in range(len(sa_pos_mask)): # 对每一层
            pos_mask = sa_pos_mask[i] > 0 # 大于0的点的mask
            gt_boxes = sa_gt_boxes[i]   # 真实的点对应框
            xyz_coords = sa_xyz_coords[i].view(-1,sa_xyz_coords[i].shape[-1])[:,1:] # [B*N,3]
            xyz_coords = xyz_coords[pos_mask].clone().detach() # 得到分数大于0的点
            offset_xyz = xyz_coords[:, 0:3] - gt_boxes[:, 0:3] # 得到和GT框中心的偏移量，也考验理解为转到局部坐标系  # （openpcdet中，x向前，y向左，z向上，x到y逆时针为正）
            offset_xyz_canical = common_utils.rotate_points_along_z(offset_xyz.unsqueeze(dim=1), -gt_boxes[:, 6]).squeeze(dim=1)#  # 上面完成平移操作后，下面完成旋转操作，使x方向朝向车头方向，y垂直于x，z向上 

            template = gt_boxes.new_tensor(([1, 1, 1], [-1, -1, -1])) / 2 # [2,3]  这个应该就是两层偏移量，就是从中心点便宜到边上，从而可以计算距离  lwh一半，所以是/2
            margin = gt_boxes[:, None, 3:6].repeat(1, 2, 1) * template[None, :, :]  # [B*N,2,3]  # 这个其实是长宽高进行扩缩操作  # chatgpt说：2是两个偏移量 可能是将目标边界框的中心点坐标根据预定义的偏移量进行平移操作，以获得调整后的中心点坐标
            distance = margin - offset_xyz_canical[:, None, :].repeat(1, 2, 1) # 长度减去对应轴的位置，  从而得到这个点对这个轴两边的长度
            distance[:, 1, :] = -1 * distance[:, 1, :] # 第二个距离值为负号，所以要乘以-1  这时候的distance[B*N,2,3]  第一个 1,3 代表着  xyz离一个边界的距离，第二个 2，3代表离另外一边的距离
            distance_min = torch.where(distance[:, 0, :] < distance[:, 1, :], distance[:, 0, :], distance[:, 1, :]) # 第一个bool类型，true选择第一个，false选择第二个
            distance_max = torch.where(distance[:, 0, :] > distance[:, 1, :], distance[:, 0, :], distance[:, 1, :])
            # 单个最小除以最大
            centerness = distance_min / distance_max
            centerness = centerness[:, 0] * centerness[:, 1] * centerness[:, 2] # 相乘
            centerness = torch.clamp(centerness, min=1e-6) # 裁剪一下最小，反手不能是负的
            centerness = torch.pow(centerness, 1/3) # 1/3次方

            centerness_mask = pos_mask.new_zeros(pos_mask.shape).float() # [B,N]
            centerness_mask[pos_mask] = centerness # 对前景点赋 这个距离值

            sa_centerness_mask.append(centerness_mask) # 加到列表中
        return sa_centerness_mask


    def get_center_box_binori_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['center_cls_labels'] > 0
        point_box_labels = self.forward_ret_dict['center_box_labels']
        point_box_preds = self.forward_ret_dict['center_box_preds']

        reg_weights = pos_mask.float()
        pos_normalizer = pos_mask.sum().float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        pred_box_xyzwhl = point_box_preds[:, :6]
        label_box_xyzwhl = point_box_labels[:, :6]

        point_loss_box_src = self.reg_loss_func(
            pred_box_xyzwhl[None, ...], label_box_xyzwhl[None, ...], weights=reg_weights[None, ...]
        )
        point_loss_xyzwhl = point_loss_box_src.sum()

        pred_ori_bin_id = point_box_preds[:, 6:6+self.box_coder.bin_size]
        pred_ori_bin_res = point_box_preds[:, 6+self.box_coder.bin_size:]

        label_ori_bin_id = point_box_labels[:, 6]
        label_ori_bin_res = point_box_labels[:, 7]
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        loss_ori_cls = criterion(pred_ori_bin_id.contiguous(), label_ori_bin_id.long().contiguous())
        loss_ori_cls = torch.sum(loss_ori_cls * reg_weights)

        label_id_one_hot = F.one_hot(label_ori_bin_id.long().contiguous(), self.box_coder.bin_size)
        pred_ori_bin_res = torch.sum(pred_ori_bin_res * label_id_one_hot.float(), dim=-1)
        loss_ori_reg = F.smooth_l1_loss(pred_ori_bin_res, label_ori_bin_res)
        loss_ori_reg = torch.sum(loss_ori_reg * reg_weights)

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        loss_ori_cls = loss_ori_cls * loss_weights_dict.get('dir_weight', 1.0)
        point_loss_box = point_loss_xyzwhl + loss_ori_reg + loss_ori_cls
        point_loss_box = point_loss_box * loss_weights_dict['point_box_weight']

        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'center_loss_box': point_loss_box.item()})
        tb_dict.update({'center_loss_box_xyzwhl': point_loss_xyzwhl.item()})
        tb_dict.update({'center_loss_box_ori_bin': loss_ori_cls.item()})
        tb_dict.update({'center_loss_box_ori_res': loss_ori_reg.item()})
        return point_loss_box, tb_dict


    def get_center_box_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['center_cls_labels'] > 0
        point_box_labels = self.forward_ret_dict['center_box_labels']
        point_box_preds = self.forward_ret_dict['center_box_preds']

        reg_weights = pos_mask.float()
        pos_normalizer = pos_mask.sum().float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        point_loss_box_src = self.reg_loss_func(
            point_box_preds[None, ...], point_box_labels[None, ...], weights=reg_weights[None, ...]
        )
        point_loss = point_loss_box_src.sum()

        point_loss_box = point_loss
        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_box = point_loss_box * loss_weights_dict['point_box_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'center_loss_box': point_loss_box.item()})
        return point_loss_box, tb_dict


    def get_corner_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['center_cls_labels'] > 0
        gt_boxes = self.forward_ret_dict['center_gt_box_of_fg_points']
        pred_boxes = self.forward_ret_dict['point_box_preds']
        pred_boxes = pred_boxes[pos_mask]
        loss_corner = loss_utils.get_corner_loss_lidar(
            pred_boxes[:, 0:7],
            gt_boxes[:, 0:7]
        )
        loss_corner = loss_corner.mean()
        loss_corner = loss_corner * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['corner_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'corner_loss_reg': loss_corner.item()})
        return loss_corner, tb_dict


    def get_iou3d_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['center_cls_labels'] > 0
        gt_boxes = self.forward_ret_dict['center_gt_box_of_fg_points']
        pred_boxes = self.forward_ret_dict['point_box_preds'].clone().detach()
        pred_boxes = pred_boxes[pos_mask]
        iou3d_targets, _ = loss_utils.generate_iou3d(pred_boxes[:, 0:7], gt_boxes[:, 0:7])

        iou3d_preds = self.forward_ret_dict['box_iou3d_preds'].squeeze(-1)
        iou3d_preds = iou3d_preds[pos_mask]

        loss_iou3d = F.smooth_l1_loss(iou3d_preds, iou3d_targets)

        loss_iou3d = loss_iou3d * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['iou3d_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'iou3d_loss_reg': loss_iou3d.item()})
        return loss_iou3d, tb_dict


    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                centers_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                centers: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                encoder_xyz: List of points_coords in SA
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                batch_cls_preds: (N1 + N2 + N3 + ..., num_class)
                point_box_preds: (N1 + N2 + N3 + ..., 7)
        """

        center_features = batch_dict['centers_features'] # [2048,521] 看第三个位置是多少点
        center_coords = batch_dict['centers']
        center_cls_preds = self.cls_center_layers(center_features)  # (total_centers, num_class)
        center_box_preds = self.box_center_layers(center_features)  # (total_centers, box_code_size)
        box_iou3d_preds = self.box_iou3d_layers(center_features) if self.box_iou3d_layers is not None else None

        ret_dict = {'center_cls_preds': center_cls_preds,
                    'center_box_preds': center_box_preds,
                    'ctr_offsets': batch_dict['ctr_offsets'],
                    'centers': batch_dict['centers'],
                    'centers_origin': batch_dict['centers_origin'],
                    'sa_ins_preds': batch_dict['sa_ins_preds'],
                    'sample_list_id':batch_dict['sample_list_id'],
                    'box_iou3d_preds': box_iou3d_preds,
                    # 'cluster_pred_class':batch_dict['cluster_feature'], # 不需要的时候注释  这个特征
                    # 'out_point_sparse_feature':batch_dict['out_point_sparse_feature']    # 这才是类别
                    }
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training or \
                self.model_cfg.LOSS_CONFIG.CORNER_LOSS_REGULARIZATION or \
                self.model_cfg.LOSS_CONFIG.CENTERNESS_REGULARIZATION or \
                self.model_cfg.LOSS_CONFIG.IOU3D_REGULARIZATION:

            point_cls_preds, point_box_preds = self.generate_predicted_boxes(
                    points=center_coords[:, 1:4],
                    point_cls_preds=center_cls_preds, point_box_preds=center_box_preds
                )

            batch_dict['batch_cls_preds'] = point_cls_preds
            batch_dict['batch_box_preds'] = point_box_preds
            batch_dict['box_iou3d_preds'] = box_iou3d_preds
            batch_dict['batch_index'] = center_coords[:,0]
            batch_dict['cls_preds_normalized'] = False

            ret_dict['point_box_preds'] = point_box_preds

        self.forward_ret_dict = ret_dict

        return batch_dict
