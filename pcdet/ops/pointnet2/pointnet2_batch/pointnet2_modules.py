from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import pointnet2_utils
from . import PointFormer

from . import semantic_view

class _PointnetSAModuleBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        self.pool_method = 'max_pool'

    def calc_square_dist(self, a, b, norm=True):
        """
        Calculating square distance between a and b
        a: [bs, n, c]
        b: [bs, m, c]
        """
        n = a.shape[1]
        m = b.shape[1]
        num_channel = a.shape[-1]
        a_square = a.unsqueeze(dim=2)  # [bs, n, 1, c]
        b_square = b.unsqueeze(dim=1)  # [bs, 1, m, c]
        a_square = torch.sum(a_square * a_square, dim=-1)  # [bs, n, 1]
        b_square = torch.sum(b_square * b_square, dim=-1)  # [bs, 1, m]
        a_square = a_square.repeat((1, 1, m))  # [bs, n, m]
        b_square = b_square.repeat((1, n, 1))  # [bs, n, m]

        coor = torch.matmul(a, b.transpose(1, 2))  # [bs, n, m]

        if norm:
            dist = a_square + b_square - 2.0 * coor  # [bs, npoint, ndataset]
            # dist = torch.sqrt(dist)
        else:
            dist = a_square + b_square - 2 * coor
            # dist = torch.sqrt(dist)
        return dist

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None, new_xyz=None) -> (torch.Tensor, torch.Tensor):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, N, C) tensor of the descriptors of the the features
        :param new_xyz:
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if new_xyz is None:
            new_xyz = pointnet2_utils.gather_operation(
                xyz_flipped,
                pointnet2_utils.farthest_point_sample(xyz, self.npoint)
            ).transpose(1, 2).contiguous() if self.npoint is not None else None

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            else:
                raise NotImplementedError

            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with multiscale grouping"""

    def __init__(self, *, npoint: int, radii: List[float], nsamples: List[int], mlps: List[List[int]], bn: bool = True,
                 use_xyz: bool = True, pool_method='max_pool'):
        """
        :param npoint: int
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        """
        super().__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlps))

        self.pool_method = pool_method


class PointnetSAModuleMSG_WithSampling_Ellipsoid_No_Global(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with specific downsampling and multiscale grouping """

    def __init__(self, *,
                 npoint_list: List[int],
                 sample_range_list: List[int],
                 sample_type_list: List[int],
                 radii: List[float],
                 nsamples: List[int],
                 mlps: List[List[int]],                 
                 use_xyz: bool = True,
                 dilated_group=False,
                 pool_method='max_pool',
                 aggregation_mlp: List[int],
                 confidence_mlp: List[int],
                 num_class):
        """
        :param npoint_list: list of int, number of samples for every sampling type
        :param sample_range_list: list of list of int, sample index range [left, right] for every sampling type
        :param sample_type_list: list of str, list of used sampling type, d-fps or f-fps
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param dilated_group: whether to use dilated group
        :param aggregation_mlp: list of int, spec aggregation mlp
        :param confidence_mlp: list of int, spec confidence mlp
        :param num_class: int, class for process
        """
        super().__init__()
        self.sample_type_list = sample_type_list
        self.sample_range_list = sample_range_list
        self.dilated_group = dilated_group

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint_list = npoint_list
        self.groupers = nn.ModuleList()
        
        self.groupers_global = nn.ModuleList()
        # self.mlps = nn.ModuleList()
        
        
        self.nsamples = nsamples
        
        
        
        # self.affine_alpha = nn.ParameterList()
        # self.affine_beta = nn.ParameterList()
        # self.score_fn = nn.ModuleList()
        
        self.point_density = nn.ModuleList()
        self.position_mlp = nn.ModuleList()
        self.Local_pointformer = nn.ModuleList()
        self.fin_conv = nn.ModuleList()
        
        # self.global_mlps =  nn.ModuleList()
        
        
        
        

        out_channels = 0
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            if self.dilated_group:
                if i == 0:
                    min_radius = 0.
                else:
                    min_radius = radii[i-1]
                self.groupers.append(
                    pointnet2_utils.QueryDilatedAndGroup(
                    # pointnet2_utils.QueryAndGroup_alone_grouped_density(
                        radius, min_radius, nsample, use_xyz=use_xyz)
                    if npoint_list is not None else pointnet2_utils.GroupAll(use_xyz)
                )
            else:
                self.groupers.append(
                    # pointnet2_utils.QueryAndGroup(
                    pointnet2_utils.QueryAndGroup_alone_grouped_density_directional(
                        radius, nsample, use_xyz=use_xyz)
                    if npoint_list is not None else pointnet2_utils.GroupAll(use_xyz)
                )
                
                
            mlp_spec = mlps[i]
            
            # 插入的模块
            self.Local_pointformer.append(PointFormer.TransformerEncoderLayerPreNorm(
                d_model=mlp_spec[0]*3, dim_feedforward=2*mlp_spec[0], dropout=0.0, nhead=4))
            
            self.position_mlp.append(nn.Sequential(
                    nn.Conv2d(9+3 , mlp_spec[0]//2, kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[0]//2),
                    nn.ReLU(),
                    nn.Conv2d(mlp_spec[0]//2 , mlp_spec[0], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[0]),
                    nn.ReLU()
                )) 
            
            # self.global_mlps.append(nn.Sequential(
            #         nn.Conv2d(mlp_spec[0]+3 , mlp_spec[0], kernel_size=1, bias=False),
            #         nn.BatchNorm2d(mlp_spec[0]),
            #         nn.ReLU(),
            #         nn.Conv2d(mlp_spec[0] , mlp_spec[0], kernel_size=1, bias=False),
            #         nn.BatchNorm2d(mlp_spec[0]),
            #         nn.ReLU()
            #     )) 
            
        
            
            # self.score_fn.append(nn.Sequential(
            #     nn.Conv2d(mlp_spec[-1], mlp_spec[-1], kernel_size=1, bias=False),
            #     nn.Softmax(dim=-1)
            # ))
            
            self.point_density.append(
                PointConvDensitySetAbstraction(radius)
   
            )
            
            self.fin_conv.append(nn.Sequential(
                nn.Conv2d(3*mlp_spec[0], 2*mlp_spec[0], kernel_size=1, bias=False),
                nn.BatchNorm2d(2*mlp_spec[0]),
                nn.ReLU(),
                nn.Conv2d(2*mlp_spec[0], mlp_spec[-1], kernel_size=1, bias=False),
                nn.BatchNorm2d( mlp_spec[-1]),
                nn.ReLU()
            ))
            
            
            # if use_xyz:
            #     mlp_spec[0] += 3

            # shared_mlps = []
            # for k in range(len(mlp_spec) - 1):
            #     shared_mlps.extend([
            #         nn.Conv2d(mlp_spec[k], mlp_spec[k + 1],
            #                   kernel_size=1, bias=False),
            #         nn.BatchNorm2d(mlp_spec[k + 1]),
            #         nn.ReLU()
            #     ])
            # self.mlps.append(nn.Sequential(*shared_mlps))
            out_channels += mlp_spec[-1]

        self.pool_method = pool_method

        # if (aggregation_mlp is not None) and (len(aggregation_mlp) != 0) and (len(self.mlps) > 0):
        if (aggregation_mlp is not None) and (len(aggregation_mlp) != 0) and (len(self.fin_conv) > 0):
            shared_mlp = []
            for k in range(len(aggregation_mlp)):
                shared_mlp.extend([
                    nn.Conv1d(out_channels,
                              aggregation_mlp[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(aggregation_mlp[k]),
                    nn.ReLU()
                ])
                out_channels = aggregation_mlp[k]
            self.aggregation_layer = nn.Sequential(*shared_mlp)
        else:
            self.aggregation_layer = None

        if (confidence_mlp is not None) and (len(confidence_mlp) != 0):
            shared_mlp = []
            for k in range(len(confidence_mlp)):
                shared_mlp.extend([
                    nn.Conv1d(out_channels,
                              confidence_mlp[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(confidence_mlp[k]),
                    nn.ReLU()
                ])
                out_channels = confidence_mlp[k]
            shared_mlp.append(
                nn.Conv1d(out_channels, num_class, kernel_size=1, bias=True),
            )
            self.confidence_layers = nn.Sequential(*shared_mlp)
        else:
            self.confidence_layers = None


    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None, cls_features: torch.Tensor = None, new_xyz=None, ctr_xyz=None):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, C, N) tensor of the descriptors of the the features
        :param cls_features: (B, N, num_class) tensor of the descriptors of the the confidence (classification) features 
        :param new_xyz: (B, M, 3) tensor of the xyz coordinates of the sampled points
        "param ctr_xyz: tensor of the xyz coordinates of the centers 
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
            cls_features: (B, npoint, num_class) tensor of confidence (classification) features
        """
        new_features_list = []
        xyz_flipped = xyz.transpose(1, 2).contiguous() 
        sampled_idx_list = []
        if ctr_xyz is None:
            last_sample_end_index = 0
            
            for i in range(len(self.sample_type_list)):
                sample_type = self.sample_type_list[i]
                sample_range = self.sample_range_list[i]
                npoint = self.npoint_list[i]

                if npoint <= 0:
                    continue
                if sample_range == -1: #全部
                    xyz_tmp = xyz[:, last_sample_end_index:, :]
                    feature_tmp = features.transpose(1, 2)[:, last_sample_end_index:, :].contiguous()  
                    cls_features_tmp = cls_features[:, last_sample_end_index:, :] if cls_features is not None else None 
                else:
                    xyz_tmp = xyz[:, last_sample_end_index:sample_range, :].contiguous()
                    feature_tmp = features.transpose(1, 2)[:, last_sample_end_index:sample_range, :]
                    cls_features_tmp = cls_features[:, last_sample_end_index:sample_range, :] if cls_features is not None else None 
                    last_sample_end_index += sample_range

                if xyz_tmp.shape[1] <= npoint: # No downsampling
                    sample_idx = torch.arange(xyz_tmp.shape[1], device=xyz_tmp.device, dtype=torch.int32) * torch.ones(xyz_tmp.shape[0], xyz_tmp.shape[1], device=xyz_tmp.device, dtype=torch.int32)

                elif ('cls' in sample_type) or ('ctr' in sample_type):
                    cls_features_max, class_pred = cls_features_tmp.max(dim=-1)
                    score_pred = torch.sigmoid(cls_features_max) # B,N
                    
                    score_picked, sample_idx = torch.topk(score_pred, npoint, dim=-1)           
                    sample_idx = sample_idx.int()

                elif 'D-FPS' in sample_type or 'DFS' in sample_type:
                    sample_idx = pointnet2_utils.furthest_point_sample(xyz_tmp.contiguous(), npoint)

                elif 'F-FPS' in sample_type or 'FFS' in sample_type:
                    features_SSD = torch.cat([xyz_tmp, feature_tmp], dim=-1)
                    features_for_fps_distance = self.calc_square_dist(features_SSD, features_SSD)
                    features_for_fps_distance = features_for_fps_distance.contiguous()
                    sample_idx = pointnet2_utils.furthest_point_sample_with_dist(features_for_fps_distance, npoint)

                elif sample_type == 'FS':
                    features_SSD = torch.cat([xyz_tmp, feature_tmp], dim=-1)
                    features_for_fps_distance = self.calc_square_dist(features_SSD, features_SSD)
                    features_for_fps_distance = features_for_fps_distance.contiguous()
                    sample_idx_1 = pointnet2_utils.furthest_point_sample_with_dist(features_for_fps_distance, npoint)
                    sample_idx_2 = pointnet2_utils.furthest_point_sample(xyz_tmp, npoint)
                    sample_idx = torch.cat([sample_idx_1, sample_idx_2], dim=-1)  # [bs, npoint * 2]
                elif 'Rand' in sample_type:
                    sample_idx = torch.randperm(xyz_tmp.shape[1],device=xyz_tmp.device)[None, :npoint].int().repeat(xyz_tmp.shape[0], 1)
                elif sample_type == 'ds_FPS' or sample_type == 'ds-FPS':
                    part_num = 4
                    xyz_div = []
                    idx_div = []
                    for i in range(len(xyz_tmp)):
                        per_xyz = xyz_tmp[i]
                        radii = per_xyz.norm(dim=-1) -5 
                        storted_radii, indince = radii.sort(dim=0, descending=False)
                        per_xyz_sorted = per_xyz[indince]
                        per_xyz_sorted_div = per_xyz_sorted.view(part_num, -1 ,3)

                        per_idx_div = indince.view(part_num,-1)
                        xyz_div.append(per_xyz_sorted_div)
                        idx_div.append(per_idx_div)
                    xyz_div = torch.cat(xyz_div ,dim=0)
                    idx_div = torch.cat(idx_div ,dim=0)
                    idx_sampled = pointnet2_utils.furthest_point_sample(xyz_div, (npoint//part_num))

                    indince_div = []
                    for idx_sampled_per, idx_per in zip(idx_sampled, idx_div):                    
                        indince_div.append(idx_per[idx_sampled_per.long()])
                    index = torch.cat(indince_div, dim=-1)
                    sample_idx = index.reshape(xyz.shape[0], npoint).int()

                elif sample_type == 'ry_FPS' or sample_type == 'ry-FPS':
                    part_num = 4
                    xyz_div = []
                    idx_div = []
                    for i in range(len(xyz_tmp)):
                        per_xyz = xyz_tmp[i]
                        ry = torch.atan(per_xyz[:,0]/per_xyz[:,1])
                        storted_ry, indince = ry.sort(dim=0, descending=False)
                        per_xyz_sorted = per_xyz[indince]
                        per_xyz_sorted_div = per_xyz_sorted.view(part_num, -1 ,3)

                        per_idx_div = indince.view(part_num,-1)
                        xyz_div.append(per_xyz_sorted_div)
                        idx_div.append(per_idx_div)
                    xyz_div = torch.cat(xyz_div ,dim=0)
                    idx_div = torch.cat(idx_div ,dim=0)
                    idx_sampled = pointnet2_utils.furthest_point_sample(xyz_div, (npoint//part_num))

                    indince_div = []
                    for idx_sampled_per, idx_per in zip(idx_sampled, idx_div):                    
                        indince_div.append(idx_per[idx_sampled_per.long()])
                    index = torch.cat(indince_div, dim=-1)

                    sample_idx = index.reshape(xyz.shape[0], npoint).int()

                sampled_idx_list.append(sample_idx)

            sampled_idx_list = torch.cat(sampled_idx_list, dim=-1) # B,N  
            new_xyz = pointnet2_utils.gather_operation(xyz_flipped, sampled_idx_list).transpose(1, 2).contiguous()  #xyz_flipped:[B,3,N]  new_xyz:B,N,3
            
            
            
            
            new_xyz_feature = pointnet2_utils.gather_operation(features, sampled_idx_list).transpose(1, 2).contiguous() # [B,N,C]

        else:
            new_xyz = ctr_xyz

        
        if len(self.groupers) > 0:
            # global_feature :[B,3+C,N,1]
            global_feature = torch.cat([new_xyz, new_xyz_feature], dim=-1).transpose(1, 2).unsqueeze(dim=-1)
            for i in range(len(self.groupers)):
                
                
                
                # new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)
                # new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
                # if self.pool_method == 'max_pool':
                #     new_features = F.max_pool2d(
                #         new_features, kernel_size=[1, new_features.size(3)]
                #     )  # (B, mlp[-1], npoint, 1)
                # elif self.pool_method == 'avg_pool':
                #     new_features = F.avg_pool2d(
                #         new_features, kernel_size=[1, new_features.size(3)]
                #     )  # (B, mlp[-1], npoint, 1)
                # else:
                #     raise NotImplementedError

                # new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
                # new_features_list.append(new_features)
                new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)
                # new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)    
                # new_features: (B, 3 + C, npoint, nsample)
                new_xyz_k = new_features[:,:3,:,:].contiguous().permute(0, 2,3, 1).contiguous()  # (B, 3, npoint, nsample)->(B, npoint, nsample, 3)
                
                grouped_density_feature = new_features[:,3:4,:,:].contiguous()  # (B,1, npoint, nsample)
                directional_vectors = new_features[:,4:7,:,:].permute(0, 2,3, 1).contiguous()
                new_xyz_K_feature  = new_features[:,7:,:,:].contiguous() # (B,C, npoint, nsample)
                
                
                
                # global_feature_k = self.global_mlps[i](global_feature).repeat(1, 1, 1, new_xyz_K_feature.size()[-1])
                
                # Two ：Attention 、Density Method
                # 1、Density Method，get ball points
                # Density:query previous layer
                
                density_scale_score = self.point_density[i](grouped_density_feature)  # sigmod: [B,1,N,K]
                # 
                # new_density_score_feature = (new_xyz_K_feature*density_scale_score).permute(0, 2,3, 1).contiguous() 
                new_density_score_feature = (new_xyz_K_feature*density_scale_score) # (B,C, npoint, nsample)
                
                # new_xyz_K_feature  = new_xyz_K_feature.permute(0, 2,3, 1).contiguous()  # (B,C, npoint, nsample)->(B, npoint, nsample,C)
                # 2、
                # 1、Attention Branch
                B= new_xyz.shape[0]
                npoint = new_xyz.shape[1]
                K_sample = self.nsamples[i]
                
                extended_coords = new_xyz.unsqueeze(-2).expand(B,npoint,K_sample,3)     # [B,N,3]->[B,N,K,3]
                # relative point position encoding
                rppe = torch.cat([
                    extended_coords,
                    new_xyz_k,
                    extended_coords - new_xyz_k,   # rppe: [B, npoint, k, 9]
                    directional_vectors
                    # dist.unsqueeze(-3) # 
                ], dim=-1)  # 
                # rppe = self.position_mlp[i](rppe.permute(0,3, 1,2).contiguous() ).permute(0,2,3, 1).contiguous()  # ->[B, npoint, k, d] 
                rppe = self.position_mlp[i](rppe.permute(0,3, 1,2).contiguous() )
                # print(rppe.shape)
                # print(new_density_score_feature.shape)
                # print(new_xyz_K_feature.shape)
                
                # input_features =  torch.cat([rppe,new_density_score_feature,new_xyz_K_feature,global_feature_k], dim=1)
                input_features =  torch.cat([rppe,new_density_score_feature,new_xyz_K_feature], dim=1)
                
                # input_features = rppe+new_density_score_feature+new_xyz_K_feature
                
                B, D, np, ns = input_features.shape
                
                # (B, C, npoint, nsample)---->(B, npoint, C, nsample)--->(B*npoint, C, nsample)---->(nsample,B*npoint, C )
                input_features = input_features.permute(0, 2, 1, 3).reshape(-1, D, ns).permute(2, 0, 1).contiguous()
                # transformer :(B, D, np, ns) 
                transformed_feats = self.Local_pointformer[i](input_features).permute(1, 2, 0).reshape(B, np, D, ns).transpose(1, 2).contiguous()
                # (B, D, np, ns)---->(B, D, np, 1)
                output_features = F.max_pool2d(transformed_feats, kernel_size=[1, ns])  # (B, C, npoint)
                # ---->(B, D, np)
                output_features = self.fin_conv[i](output_features).squeeze(-1) # 

                # return new_xyz, output_features
                new_features_list.append(output_features)
                    
                    
                    
                

            new_features = torch.cat(new_features_list, dim=1)

            if self.aggregation_layer is not None:
                new_features = self.aggregation_layer(new_features)
        else:
            new_features = pointnet2_utils.gather_operation(features, sampled_idx_list).contiguous()

        if self.confidence_layers is not None:
            cls_features = self.confidence_layers(new_features).transpose(1, 2)
        else:
            cls_features = None

        return new_xyz, new_features, cls_features,sampled_idx_list

class PointnetSAModuleMSG_WithSampling_Ellipsoid(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with specific downsampling and multiscale grouping """

    def __init__(self, *,
                 npoint_list: List[int],
                 sample_range_list: List[int],
                 sample_type_list: List[int],
                 radii: List[float],
                 nsamples: List[int],
                 mlps: List[List[int]],                 
                 use_xyz: bool = True,
                 dilated_group=False,
                 pool_method='max_pool',
                 aggregation_mlp: List[int],
                 confidence_mlp: List[int],
                 num_class):
        """
        :param npoint_list: list of int, number of samples for every sampling type
        :param sample_range_list: list of list of int, sample index range [left, right] for every sampling type
        :param sample_type_list: list of str, list of used sampling type, d-fps or f-fps
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param dilated_group: whether to use dilated group
        :param aggregation_mlp: list of int, spec aggregation mlp
        :param confidence_mlp: list of int, spec confidence mlp
        :param num_class: int, class for process
        """
        super().__init__()
        self.sample_type_list = sample_type_list
        self.sample_range_list = sample_range_list
        self.dilated_group = dilated_group

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint_list = npoint_list
        self.groupers = nn.ModuleList()
        
        self.groupers_global = nn.ModuleList()
        # self.mlps = nn.ModuleList()
        
        
        self.nsamples = nsamples
        
        
        
        # self.affine_alpha = nn.ParameterList()
        # self.affine_beta = nn.ParameterList()
        # self.score_fn = nn.ModuleList()
        
        self.point_density = nn.ModuleList()
        self.position_mlp = nn.ModuleList()
        self.Local_pointformer = nn.ModuleList()
        self.fin_conv = nn.ModuleList()
        
        self.global_mlps =  nn.ModuleList()
        
        
        
        

        out_channels = 0
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            if self.dilated_group:
                if i == 0:
                    min_radius = 0.
                else:
                    min_radius = radii[i-1]
                self.groupers.append(
                    pointnet2_utils.QueryDilatedAndGroup(
                    # pointnet2_utils.QueryAndGroup_alone_grouped_density(
                        radius, min_radius, nsample, use_xyz=use_xyz)
                    if npoint_list is not None else pointnet2_utils.GroupAll(use_xyz)
                )
            else:
                self.groupers.append(
                    # pointnet2_utils.QueryAndGroup(
                    pointnet2_utils.QueryAndGroup_alone_grouped_density_directional(
                        radius, nsample, use_xyz=use_xyz)
                    if npoint_list is not None else pointnet2_utils.GroupAll(use_xyz)
                )
                
                
            mlp_spec = mlps[i]
            
            # 
            self.Local_pointformer.append(PointFormer.TransformerEncoderLayerPreNorm(
                d_model=mlp_spec[0]*4, dim_feedforward=2*mlp_spec[0], dropout=0.0, nhead=4))
            
            self.position_mlp.append(nn.Sequential(
                    nn.Conv2d(9+3 , mlp_spec[0]//2, kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[0]//2),
                    nn.ReLU(),
                    nn.Conv2d(mlp_spec[0]//2 , mlp_spec[0], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[0]),
                    nn.ReLU()
                )) 
            
            self.global_mlps.append(nn.Sequential(
                    nn.Conv2d(mlp_spec[0]+3 , mlp_spec[0], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[0]),
                    nn.ReLU(),
                    nn.Conv2d(mlp_spec[0] , mlp_spec[0], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[0]),
                    nn.ReLU()
                )) 
            

            
            # self.score_fn.append(nn.Sequential(
            #     nn.Conv2d(mlp_spec[-1], mlp_spec[-1], kernel_size=1, bias=False),
            #     nn.Softmax(dim=-1)
            # ))
            
            self.point_density.append(
                PointConvDensitySetAbstraction(radius)
   
            )
            
            self.fin_conv.append(nn.Sequential(
                nn.Conv2d(4*mlp_spec[0], 2*mlp_spec[0], kernel_size=1, bias=False),
                nn.BatchNorm2d(2*mlp_spec[0]),
                nn.ReLU(),
                nn.Conv2d(2*mlp_spec[0], mlp_spec[-1], kernel_size=1, bias=False),
                nn.BatchNorm2d( mlp_spec[-1]),
                nn.ReLU()
            ))
            
            
            # if use_xyz:
            #     mlp_spec[0] += 3

            # shared_mlps = []
            # for k in range(len(mlp_spec) - 1):
            #     shared_mlps.extend([
            #         nn.Conv2d(mlp_spec[k], mlp_spec[k + 1],
            #                   kernel_size=1, bias=False),
            #         nn.BatchNorm2d(mlp_spec[k + 1]),
            #         nn.ReLU()
            #     ])
            # self.mlps.append(nn.Sequential(*shared_mlps))
            out_channels += mlp_spec[-1]

        self.pool_method = pool_method

        # if (aggregation_mlp is not None) and (len(aggregation_mlp) != 0) and (len(self.mlps) > 0):
        if (aggregation_mlp is not None) and (len(aggregation_mlp) != 0) and (len(self.fin_conv) > 0):
            shared_mlp = []
            for k in range(len(aggregation_mlp)):
                shared_mlp.extend([
                    nn.Conv1d(out_channels,
                              aggregation_mlp[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(aggregation_mlp[k]),
                    nn.ReLU()
                ])
                out_channels = aggregation_mlp[k]
            self.aggregation_layer = nn.Sequential(*shared_mlp)
        else:
            self.aggregation_layer = None

        if (confidence_mlp is not None) and (len(confidence_mlp) != 0):
            shared_mlp = []
            for k in range(len(confidence_mlp)):
                shared_mlp.extend([
                    nn.Conv1d(out_channels,
                              confidence_mlp[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(confidence_mlp[k]),
                    nn.ReLU()
                ])
                out_channels = confidence_mlp[k]
            shared_mlp.append(
                nn.Conv1d(out_channels, num_class, kernel_size=1, bias=True),
            )
            self.confidence_layers = nn.Sequential(*shared_mlp)
        else:
            self.confidence_layers = None


    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None, cls_features: torch.Tensor = None, new_xyz=None, ctr_xyz=None):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, C, N) tensor of the descriptors of the the features
        :param cls_features: (B, N, num_class) tensor of the descriptors of the the confidence (classification) features 
        :param new_xyz: (B, M, 3) tensor of the xyz coordinates of the sampled points
        "param ctr_xyz: tensor of the xyz coordinates of the centers 
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
            cls_features: (B, npoint, num_class) tensor of confidence (classification) features
        """
        new_features_list = []
        xyz_flipped = xyz.transpose(1, 2).contiguous() 
        sampled_idx_list = []
        if ctr_xyz is None:
            last_sample_end_index = 0
            
            for i in range(len(self.sample_type_list)):
                sample_type = self.sample_type_list[i]
                sample_range = self.sample_range_list[i]
                npoint = self.npoint_list[i]

                if npoint <= 0:
                    continue
                if sample_range == -1: #全部
                    xyz_tmp = xyz[:, last_sample_end_index:, :]
                    feature_tmp = features.transpose(1, 2)[:, last_sample_end_index:, :].contiguous()  
                    cls_features_tmp = cls_features[:, last_sample_end_index:, :] if cls_features is not None else None 
                else:
                    xyz_tmp = xyz[:, last_sample_end_index:sample_range, :].contiguous()
                    feature_tmp = features.transpose(1, 2)[:, last_sample_end_index:sample_range, :]
                    cls_features_tmp = cls_features[:, last_sample_end_index:sample_range, :] if cls_features is not None else None 
                    last_sample_end_index += sample_range

                if xyz_tmp.shape[1] <= npoint: # No downsampling
                    sample_idx = torch.arange(xyz_tmp.shape[1], device=xyz_tmp.device, dtype=torch.int32) * torch.ones(xyz_tmp.shape[0], xyz_tmp.shape[1], device=xyz_tmp.device, dtype=torch.int32)

                elif ('cls' in sample_type) or ('ctr' in sample_type):
                    cls_features_max, class_pred = cls_features_tmp.max(dim=-1)
                    score_pred = torch.sigmoid(cls_features_max) # B,N
                    
        
                    
                    
                    
                    score_picked, sample_idx = torch.topk(score_pred, npoint, dim=-1)           
                    sample_idx = sample_idx.int()

                elif 'D-FPS' in sample_type or 'DFS' in sample_type:
                    sample_idx = pointnet2_utils.furthest_point_sample(xyz_tmp.contiguous(), npoint)

                elif 'F-FPS' in sample_type or 'FFS' in sample_type:
                    features_SSD = torch.cat([xyz_tmp, feature_tmp], dim=-1)
                    features_for_fps_distance = self.calc_square_dist(features_SSD, features_SSD)
                    features_for_fps_distance = features_for_fps_distance.contiguous()
                    sample_idx = pointnet2_utils.furthest_point_sample_with_dist(features_for_fps_distance, npoint)

                elif sample_type == 'FS':
                    features_SSD = torch.cat([xyz_tmp, feature_tmp], dim=-1)
                    features_for_fps_distance = self.calc_square_dist(features_SSD, features_SSD)
                    features_for_fps_distance = features_for_fps_distance.contiguous()
                    sample_idx_1 = pointnet2_utils.furthest_point_sample_with_dist(features_for_fps_distance, npoint)
                    sample_idx_2 = pointnet2_utils.furthest_point_sample(xyz_tmp, npoint)
                    sample_idx = torch.cat([sample_idx_1, sample_idx_2], dim=-1)  # [bs, npoint * 2]
                elif 'Rand' in sample_type:
                    sample_idx = torch.randperm(xyz_tmp.shape[1],device=xyz_tmp.device)[None, :npoint].int().repeat(xyz_tmp.shape[0], 1)
                elif sample_type == 'ds_FPS' or sample_type == 'ds-FPS':
                    part_num = 4
                    xyz_div = []
                    idx_div = []
                    for i in range(len(xyz_tmp)):
                        per_xyz = xyz_tmp[i]
                        radii = per_xyz.norm(dim=-1) -5 
                        storted_radii, indince = radii.sort(dim=0, descending=False)
                        per_xyz_sorted = per_xyz[indince]
                        per_xyz_sorted_div = per_xyz_sorted.view(part_num, -1 ,3)

                        per_idx_div = indince.view(part_num,-1)
                        xyz_div.append(per_xyz_sorted_div)
                        idx_div.append(per_idx_div)
                    xyz_div = torch.cat(xyz_div ,dim=0)
                    idx_div = torch.cat(idx_div ,dim=0)
                    idx_sampled = pointnet2_utils.furthest_point_sample(xyz_div, (npoint//part_num))

                    indince_div = []
                    for idx_sampled_per, idx_per in zip(idx_sampled, idx_div):                    
                        indince_div.append(idx_per[idx_sampled_per.long()])
                    index = torch.cat(indince_div, dim=-1)
                    sample_idx = index.reshape(xyz.shape[0], npoint).int()

                elif sample_type == 'ry_FPS' or sample_type == 'ry-FPS':
                    part_num = 4
                    xyz_div = []
                    idx_div = []
                    for i in range(len(xyz_tmp)):
                        per_xyz = xyz_tmp[i]
                        ry = torch.atan(per_xyz[:,0]/per_xyz[:,1])
                        storted_ry, indince = ry.sort(dim=0, descending=False)
                        per_xyz_sorted = per_xyz[indince]
                        per_xyz_sorted_div = per_xyz_sorted.view(part_num, -1 ,3)

                        per_idx_div = indince.view(part_num,-1)
                        xyz_div.append(per_xyz_sorted_div)
                        idx_div.append(per_idx_div)
                    xyz_div = torch.cat(xyz_div ,dim=0)
                    idx_div = torch.cat(idx_div ,dim=0)
                    idx_sampled = pointnet2_utils.furthest_point_sample(xyz_div, (npoint//part_num))

                    indince_div = []
                    for idx_sampled_per, idx_per in zip(idx_sampled, idx_div):                    
                        indince_div.append(idx_per[idx_sampled_per.long()])
                    index = torch.cat(indince_div, dim=-1)

                    sample_idx = index.reshape(xyz.shape[0], npoint).int()

                sampled_idx_list.append(sample_idx)

            sampled_idx_list = torch.cat(sampled_idx_list, dim=-1) # B,N  
            new_xyz = pointnet2_utils.gather_operation(xyz_flipped, sampled_idx_list).transpose(1, 2).contiguous()  #xyz_flipped:[B,3,N]  new_xyz:B,N,3
            

            
            
            
            new_xyz_feature = pointnet2_utils.gather_operation(features, sampled_idx_list).transpose(1, 2).contiguous() # [B,N,C]

        else:
            new_xyz = ctr_xyz

        
        if len(self.groupers) > 0:
            # global_feature :[B,3+C,N,1]
            global_feature = torch.cat([new_xyz, new_xyz_feature], dim=-1).transpose(1, 2).unsqueeze(dim=-1)
            for i in range(len(self.groupers)):
                
                
                
                # new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)
                # new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
                # if self.pool_method == 'max_pool':
                #     new_features = F.max_pool2d(
                #         new_features, kernel_size=[1, new_features.size(3)]
                #     )  # (B, mlp[-1], npoint, 1)
                # elif self.pool_method == 'avg_pool':
                #     new_features = F.avg_pool2d(
                #         new_features, kernel_size=[1, new_features.size(3)]
                #     )  # (B, mlp[-1], npoint, 1)
                # else:
                #     raise NotImplementedError

                # new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
                # new_features_list.append(new_features)
                new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)
                # new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)    
                # new_features: (B, 3 + C, npoint, nsample)
                new_xyz_k = new_features[:,:3,:,:].contiguous().permute(0, 2,3, 1).contiguous()  # (B, 3, npoint, nsample)->(B, npoint, nsample, 3)
                
                grouped_density_feature = new_features[:,3:4,:,:].contiguous()  # (B,1, npoint, nsample)
                directional_vectors = new_features[:,4:7,:,:].permute(0, 2,3, 1).contiguous()
                new_xyz_K_feature  = new_features[:,7:,:,:].contiguous() # (B,C, npoint, nsample)
                
                
                
                global_feature_k = self.global_mlps[i](global_feature).repeat(1, 1, 1, new_xyz_K_feature.size()[-1])
                
                # Two ：Attention 、Density Method
                # 1、Density Method，get ball points
                # Density:query previous layer
                
                density_scale_score = self.point_density[i](grouped_density_feature)  # sigmod: [B,1,N,K]
                # 
                # new_density_score_feature = (new_xyz_K_feature*density_scale_score).permute(0, 2,3, 1).contiguous() 
                new_density_score_feature = (new_xyz_K_feature*density_scale_score) # (B,C, npoint, nsample)
                
                # new_xyz_K_feature  = new_xyz_K_feature.permute(0, 2,3, 1).contiguous()  # (B,C, npoint, nsample)->(B, npoint, nsample,C)
                # 2、
                # 1、Attention  branch
                B= new_xyz.shape[0]
                npoint = new_xyz.shape[1]
                K_sample = self.nsamples[i]
                
                extended_coords = new_xyz.unsqueeze(-2).expand(B,npoint,K_sample,3)     # [B,N,3]->[B,N,K,3]
                # relative point position encoding
                rppe = torch.cat([
                    extended_coords,
                    new_xyz_k,
                    extended_coords - new_xyz_k,   # rppe: [B, npoint, k, 9]
                    directional_vectors
                    # dist.unsqueeze(-3) # 
                ], dim=-1)  # 
                # rppe = self.position_mlp[i](rppe.permute(0,3, 1,2).contiguous() ).permute(0,2,3, 1).contiguous()  # ->[B, npoint, k, d] 
                rppe = self.position_mlp[i](rppe.permute(0,3, 1,2).contiguous() )
                # print(rppe.shape)
                # print(new_density_score_feature.shape)
                # print(new_xyz_K_feature.shape)
                
                input_features =  torch.cat([rppe,new_density_score_feature,new_xyz_K_feature,global_feature_k], dim=1)
                
                # input_features = rppe+new_density_score_feature+new_xyz_K_feature
                
                B, D, np, ns = input_features.shape
                
                # (B, C, npoint, nsample)---->(B, npoint, C, nsample)--->(B*npoint, C, nsample)---->(nsample,B*npoint, C )
                input_features = input_features.permute(0, 2, 1, 3).reshape(-1, D, ns).permute(2, 0, 1).contiguous()
                # transformer :(B, D, np, ns) 最终还是恢复成了这个
                transformed_feats = self.Local_pointformer[i](input_features).permute(1, 2, 0).reshape(B, np, D, ns).transpose(1, 2).contiguous()
                # (B, D, np, ns)---->(B, D, np, 1)
                output_features = F.max_pool2d(transformed_feats, kernel_size=[1, ns])  # (B, C, npoint)
                # ---->(B, D, np)
                output_features = self.fin_conv[i](output_features).squeeze(-1) # MLP之后去掉 最后的值

                # return new_xyz, output_features
                new_features_list.append(output_features)
                    
                    
                    
                

            new_features = torch.cat(new_features_list, dim=1)

            if self.aggregation_layer is not None:
                new_features = self.aggregation_layer(new_features)
        else:
            new_features = pointnet2_utils.gather_operation(features, sampled_idx_list).contiguous()

        if self.confidence_layers is not None:
            cls_features = self.confidence_layers(new_features).transpose(1, 2)
        else:
            cls_features = None

        return new_xyz, new_features, cls_features,sampled_idx_list



class DensityNet(nn.Module):
    def __init__(self, hidden_unit = [16, 8]):
        super(DensityNet, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList() 

        self.mlp_convs.append(nn.Conv2d(1, hidden_unit[0], 1))
        self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
        for i in range(1, len(hidden_unit)):
            self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
        self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], 1, 1))
        self.mlp_bns.append(nn.BatchNorm2d(1))

    def forward(self, density_scale):
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            density_scale =  bn(conv(density_scale))
            if i == len(self.mlp_convs):
                density_scale = F.sigmoid(density_scale)
            else:
                density_scale = F.relu(density_scale)
        
        return density_scale

class PointConvDensitySetAbstraction(nn.Module):
    def __init__(self, bandwidth):
        super(PointConvDensitySetAbstraction, self).__init__()
        self.densitynet = DensityNet()
        self.bandwidth = bandwidth
    
    def forward(self, grouped_density):
        """
        Input:
           
            new_xyz_k:  [B, N,k,3 ]
            xyz:         [B, N, 3] 
            
        Return:
         
        """

        inverse_max_density = grouped_density.max(dim = 3, keepdim=True)[0] # 
        density_scale = grouped_density / inverse_max_density   # B,N,K,1
     
        density_scale = self.densitynet(density_scale)  # output: B,1,N,K
        
        
        return density_scale
        


class CBAM(nn.Module):
    def __init__(self, cn):
        super(CBAM, self).__init__()
        # self.MLP1 = nn.Sequential(
        #             nn.Conv1d(cn, cn,kernel_size=1, bias=False),
        #             nn.BatchNorm1d(cn),
        #             nn.ReLU())
        
        # self.MLP1 = nn.Conv1d(cn, cn,kernel_size=1, bias=False)
        
  

        # self.sig1 = torch.nn.Sigmoid()
        self.sig2 = torch.nn.Sigmoid()
        self.conv_layer = nn.Conv1d(2, 1,kernel_size=1, bias=False)

    # Proposal Feature  B,C,N
    def forward(self, input):
        # MP1 = F.max_pool1d(input, kernel_size=input.size(2))
        # AP1 = F.avg_pool1d(input, kernel_size=input.size(2))

        # MP1 = self.MLP1(MP1)
        # AP1 = self.MLP1(AP1)

        # weight_1 = MP1+AP1

        # input = input*self.sig1(weight_1)

        MP2 = F.max_pool1d(input.transpose(1, 2).contiguous(), kernel_size=input.size(1))
        AP2 = F.avg_pool1d(input.transpose(1, 2).contiguous(), kernel_size=input.size(1))
        MP_AP = torch.cat([MP2, AP2], 2)
        MP_AP = MP_AP.transpose(1, 2).contiguous()
        MP_AP = self.conv_layer(MP_AP)
        weight_2 = self.sig2(MP_AP)
        input = input*weight_2

        return input

class PointnetSAModuleMSG_WithSampling_Proposal_Aware(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with specific downsampling and multiscale grouping """

    def __init__(self, *,
                 npoint_list: List[int],
                 sample_range_list: List[int],
                 sample_type_list: List[int],
                 radii: List[float],
                 nsamples: List[int],
                 mlps: List[List[int]],                 
                 use_xyz: bool = True,
                 dilated_group=False,
                 pool_method='max_pool',
                 aggregation_mlp: List[int],
                 confidence_mlp: List[int],
                 num_class):
        """
        :param npoint_list: list of int, number of samples for every sampling type
        :param sample_range_list: list of list of int, sample index range [left, right] for every sampling type
        :param sample_type_list: list of str, list of used sampling type, d-fps or f-fps
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param dilated_group: whether to use dilated group
        :param aggregation_mlp: list of int, spec aggregation mlp
        :param confidence_mlp: list of int, spec confidence mlp
        :param num_class: int, class for process
        """
        super().__init__()
        self.sample_type_list = sample_type_list
        self.sample_range_list = sample_range_list
        self.dilated_group = dilated_group

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint_list = npoint_list
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()

        out_channels = 0
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            if self.dilated_group:
                if i == 0:
                    min_radius = 0.
                else:
                    min_radius = radii[i-1]
                self.groupers.append(
                    pointnet2_utils.QueryDilatedAndGroup(
                        radius, min_radius, nsample, use_xyz=use_xyz)
                    if npoint_list is not None else pointnet2_utils.GroupAll(use_xyz)
                )
            else:
                self.groupers.append(
                    pointnet2_utils.QueryAndGroup(
                        radius, nsample, use_xyz=use_xyz)
                    if npoint_list is not None else pointnet2_utils.GroupAll(use_xyz)
                )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1],
                              kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlps))
            out_channels += mlp_spec[-1]

        self.pool_method = pool_method

        if (aggregation_mlp is not None) and (len(aggregation_mlp) != 0) and (len(self.mlps) > 0):
            shared_mlp = []
            for k in range(len(aggregation_mlp)):
                shared_mlp.extend([
                    nn.Conv1d(out_channels,
                              aggregation_mlp[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(aggregation_mlp[k]),
                    nn.ReLU()
                ])
                out_channels = aggregation_mlp[k]
            self.aggregation_layer = nn.Sequential(*shared_mlp)
        else:
            self.aggregation_layer = None
            
        self.cbam = CBAM(out_channels)

        if (confidence_mlp is not None) and (len(confidence_mlp) != 0):
            shared_mlp = []
            for k in range(len(confidence_mlp)):
                shared_mlp.extend([
                    nn.Conv1d(out_channels,
                              confidence_mlp[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(confidence_mlp[k]),
                    nn.ReLU()
                ])
                out_channels = confidence_mlp[k]
            shared_mlp.append(
                nn.Conv1d(out_channels, num_class, kernel_size=1, bias=True),
            )
            self.confidence_layers = nn.Sequential(*shared_mlp)
        else:
            self.confidence_layers = None


    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None, cls_features: torch.Tensor = None, new_xyz=None, ctr_xyz=None):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, C, N) tensor of the descriptors of the the features
        :param cls_features: (B, N, num_class) tensor of the descriptors of the the confidence (classification) features 
        :param new_xyz: (B, M, 3) tensor of the xyz coordinates of the sampled points
        "param ctr_xyz: tensor of the xyz coordinates of the centers 
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
            cls_features: (B, npoint, num_class) tensor of confidence (classification) features
        """
        new_features_list = []
        xyz_flipped = xyz.transpose(1, 2).contiguous() 
        sampled_idx_list = []
        if ctr_xyz is None:
            last_sample_end_index = 0
            
            for i in range(len(self.sample_type_list)):
                sample_type = self.sample_type_list[i]
                sample_range = self.sample_range_list[i]
                npoint = self.npoint_list[i]

                if npoint <= 0:
                    continue
                if sample_range == -1: #全部
                    xyz_tmp = xyz[:, last_sample_end_index:, :]
                    feature_tmp = features.transpose(1, 2)[:, last_sample_end_index:, :].contiguous()  
                    cls_features_tmp = cls_features[:, last_sample_end_index:, :] if cls_features is not None else None 
                else:
                    xyz_tmp = xyz[:, last_sample_end_index:sample_range, :].contiguous()
                    feature_tmp = features.transpose(1, 2)[:, last_sample_end_index:sample_range, :]
                    cls_features_tmp = cls_features[:, last_sample_end_index:sample_range, :] if cls_features is not None else None 
                    last_sample_end_index += sample_range

                if xyz_tmp.shape[1] <= npoint: # No downsampling
                    sample_idx = torch.arange(xyz_tmp.shape[1], device=xyz_tmp.device, dtype=torch.int32) * torch.ones(xyz_tmp.shape[0], xyz_tmp.shape[1], device=xyz_tmp.device, dtype=torch.int32)

                elif ('cls' in sample_type) or ('ctr' in sample_type):
                    cls_features_max, class_pred = cls_features_tmp.max(dim=-1)
                    score_pred = torch.sigmoid(cls_features_max) # B,N
                    
                    # input_semantic_score = score_pred[0,:].unsqueeze(-1)
                    # print("xyz的尺寸:",xyz.shape,"  语义分数：",input_semantic_score.max())
                    # semantic_view.get_vis(xyz[0,:,:],input_semantic_score)
                    
                    
                    score_picked, sample_idx = torch.topk(score_pred, npoint, dim=-1)           
                    sample_idx = sample_idx.int()

                elif 'D-FPS' in sample_type or 'DFS' in sample_type:
                    sample_idx = pointnet2_utils.furthest_point_sample(xyz_tmp.contiguous(), npoint)

                elif 'F-FPS' in sample_type or 'FFS' in sample_type:
                    features_SSD = torch.cat([xyz_tmp, feature_tmp], dim=-1)
                    features_for_fps_distance = self.calc_square_dist(features_SSD, features_SSD)
                    features_for_fps_distance = features_for_fps_distance.contiguous()
                    sample_idx = pointnet2_utils.furthest_point_sample_with_dist(features_for_fps_distance, npoint)

                elif sample_type == 'FS':
                    features_SSD = torch.cat([xyz_tmp, feature_tmp], dim=-1)
                    features_for_fps_distance = self.calc_square_dist(features_SSD, features_SSD)
                    features_for_fps_distance = features_for_fps_distance.contiguous()
                    sample_idx_1 = pointnet2_utils.furthest_point_sample_with_dist(features_for_fps_distance, npoint)
                    sample_idx_2 = pointnet2_utils.furthest_point_sample(xyz_tmp, npoint)
                    sample_idx = torch.cat([sample_idx_1, sample_idx_2], dim=-1)  # [bs, npoint * 2]
                elif 'Rand' in sample_type:
                    sample_idx = torch.randperm(xyz_tmp.shape[1],device=xyz_tmp.device)[None, :npoint].int().repeat(xyz_tmp.shape[0], 1)
                elif sample_type == 'ds_FPS' or sample_type == 'ds-FPS':
                    part_num = 4
                    xyz_div = []
                    idx_div = []
                    for i in range(len(xyz_tmp)):
                        per_xyz = xyz_tmp[i]
                        radii = per_xyz.norm(dim=-1) -5 
                        storted_radii, indince = radii.sort(dim=0, descending=False)
                        per_xyz_sorted = per_xyz[indince]
                        per_xyz_sorted_div = per_xyz_sorted.view(part_num, -1 ,3)

                        per_idx_div = indince.view(part_num,-1)
                        xyz_div.append(per_xyz_sorted_div)
                        idx_div.append(per_idx_div)
                    xyz_div = torch.cat(xyz_div ,dim=0)
                    idx_div = torch.cat(idx_div ,dim=0)
                    idx_sampled = pointnet2_utils.furthest_point_sample(xyz_div, (npoint//part_num))

                    indince_div = []
                    for idx_sampled_per, idx_per in zip(idx_sampled, idx_div):                    
                        indince_div.append(idx_per[idx_sampled_per.long()])
                    index = torch.cat(indince_div, dim=-1)
                    sample_idx = index.reshape(xyz.shape[0], npoint).int()

                elif sample_type == 'ry_FPS' or sample_type == 'ry-FPS':
                    part_num = 4
                    xyz_div = []
                    idx_div = []
                    for i in range(len(xyz_tmp)):
                        per_xyz = xyz_tmp[i]
                        ry = torch.atan(per_xyz[:,0]/per_xyz[:,1])
                        storted_ry, indince = ry.sort(dim=0, descending=False)
                        per_xyz_sorted = per_xyz[indince]
                        per_xyz_sorted_div = per_xyz_sorted.view(part_num, -1 ,3)

                        per_idx_div = indince.view(part_num,-1)
                        xyz_div.append(per_xyz_sorted_div)
                        idx_div.append(per_idx_div)
                    xyz_div = torch.cat(xyz_div ,dim=0)
                    idx_div = torch.cat(idx_div ,dim=0)
                    idx_sampled = pointnet2_utils.furthest_point_sample(xyz_div, (npoint//part_num))

                    indince_div = []
                    for idx_sampled_per, idx_per in zip(idx_sampled, idx_div):                    
                        indince_div.append(idx_per[idx_sampled_per.long()])
                    index = torch.cat(indince_div, dim=-1)

                    sample_idx = index.reshape(xyz.shape[0], npoint).int()

                sampled_idx_list.append(sample_idx)

            sampled_idx_list = torch.cat(sampled_idx_list, dim=-1) 
            new_xyz = pointnet2_utils.gather_operation(xyz_flipped, sampled_idx_list).transpose(1, 2).contiguous()
            
            # semantic_view.get_vis(new_xyz[0,:,:],torch.ones_like(new_xyz[0,:,:]))
            

        else:
            new_xyz = ctr_xyz

        if len(self.groupers) > 0:
            for i in range(len(self.groupers)):
                new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)
                new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
                if self.pool_method == 'max_pool':
                    new_features = F.max_pool2d(
                        new_features, kernel_size=[1, new_features.size(3)]
                    )  # (B, mlp[-1], npoint, 1)
                elif self.pool_method == 'avg_pool':
                    new_features = F.avg_pool2d(
                        new_features, kernel_size=[1, new_features.size(3)]
                    )  # (B, mlp[-1], npoint, 1)
                else:
                    raise NotImplementedError

                new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
                # 
                new_features_list.append(new_features)

            new_features = torch.cat(new_features_list, dim=1)

            if self.aggregation_layer is not None:
                new_features = self.aggregation_layer(new_features)
        else:
            new_features = pointnet2_utils.gather_operation(features, sampled_idx_list).contiguous()

        new_features = self.cbam(new_features)
        
        if self.confidence_layers is not None:
            cls_features = self.confidence_layers(new_features).transpose(1, 2)            
        else:
            cls_features = None

        return new_xyz, new_features, cls_features,sampled_idx_list



class EncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead):
        super(EncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, weight=None, x_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        residual = x
        x = self.norm1(x)
        query, key, value = x, x, x

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        # value = (self.v_proj(value) * weight.transpose(1, 2)).view(bs, -1, self.nhead, self.dim)
        value = (self.v_proj(value) ).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=x_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C] 
        message = message + residual

        # feed-forward network
        residual = message
        message = self.norm2(message)
        message = self.mlp(message)

        # Renewed Object Candidates
        return residual + message


class FullAttention(nn.Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        if kv_mask is not None:
            QK.masked_fill_(~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float('-inf'))

        # Compute the attention and the weighted average
        softmax_temp = 1. / queries.size(3)**.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        return queried_values.contiguous()


class PointnetSAModuleMSG_WithSampling(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with specific downsampling and multiscale grouping """

    def __init__(self, *,
                 npoint_list: List[int],
                 sample_range_list: List[int],
                 sample_type_list: List[int],
                 radii: List[float],
                 nsamples: List[int],
                 mlps: List[List[int]],                 
                 use_xyz: bool = True,
                 dilated_group=False,
                 pool_method='max_pool',
                 aggregation_mlp: List[int],
                 confidence_mlp: List[int],
                 num_class):
        """
        :param npoint_list: list of int, number of samples for every sampling type
        :param sample_range_list: list of list of int, sample index range [left, right] for every sampling type
        :param sample_type_list: list of str, list of used sampling type, d-fps or f-fps
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param dilated_group: whether to use dilated group
        :param aggregation_mlp: list of int, spec aggregation mlp
        :param confidence_mlp: list of int, spec confidence mlp
        :param num_class: int, class for process
        """
        super().__init__()
        self.sample_type_list = sample_type_list
        self.sample_range_list = sample_range_list
        self.dilated_group = dilated_group

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint_list = npoint_list
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()

        out_channels = 0
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            if self.dilated_group:
                if i == 0:
                    min_radius = 0.
                else:
                    min_radius = radii[i-1]
                self.groupers.append(
                    pointnet2_utils.QueryDilatedAndGroup(
                        radius, min_radius, nsample, use_xyz=use_xyz)
                    if npoint_list is not None else pointnet2_utils.GroupAll(use_xyz)
                )
            else:
                self.groupers.append(
                    pointnet2_utils.QueryAndGroup(
                        radius, nsample, use_xyz=use_xyz)
                    if npoint_list is not None else pointnet2_utils.GroupAll(use_xyz)
                )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1],
                              kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlps))
            out_channels += mlp_spec[-1]

        self.pool_method = pool_method

        if (aggregation_mlp is not None) and (len(aggregation_mlp) != 0) and (len(self.mlps) > 0):
            shared_mlp = []
            for k in range(len(aggregation_mlp)):
                shared_mlp.extend([
                    nn.Conv1d(out_channels,
                              aggregation_mlp[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(aggregation_mlp[k]),
                    nn.ReLU()
                ])
                out_channels = aggregation_mlp[k]
            self.aggregation_layer = nn.Sequential(*shared_mlp)
        else:
            self.aggregation_layer = None

        if (confidence_mlp is not None) and (len(confidence_mlp) != 0):
            shared_mlp = []
            for k in range(len(confidence_mlp)):
                shared_mlp.extend([
                    nn.Conv1d(out_channels,
                              confidence_mlp[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(confidence_mlp[k]),
                    nn.ReLU()
                ])
                out_channels = confidence_mlp[k]
            shared_mlp.append(
                nn.Conv1d(out_channels, num_class, kernel_size=1, bias=True),
            )
            self.confidence_layers = nn.Sequential(*shared_mlp)
        else:
            self.confidence_layers = None


    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None, cls_features: torch.Tensor = None, new_xyz=None, ctr_xyz=None):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, C, N) tensor of the descriptors of the the features
        :param cls_features: (B, N, num_class) tensor of the descriptors of the the confidence (classification) features 
        :param new_xyz: (B, M, 3) tensor of the xyz coordinates of the sampled points
        "param ctr_xyz: tensor of the xyz coordinates of the centers 
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
            cls_features: (B, npoint, num_class) tensor of confidence (classification) features
        """
        new_features_list = []
        xyz_flipped = xyz.transpose(1, 2).contiguous() 
        sampled_idx_list = []
        if ctr_xyz is None:
            last_sample_end_index = 0
            
            for i in range(len(self.sample_type_list)):
                sample_type = self.sample_type_list[i]
                sample_range = self.sample_range_list[i]
                npoint = self.npoint_list[i]

                if npoint <= 0:
                    continue
                if sample_range == -1: #全部
                    xyz_tmp = xyz[:, last_sample_end_index:, :]
                    feature_tmp = features.transpose(1, 2)[:, last_sample_end_index:, :].contiguous()  
                    cls_features_tmp = cls_features[:, last_sample_end_index:, :] if cls_features is not None else None 
                else:
                    xyz_tmp = xyz[:, last_sample_end_index:sample_range, :].contiguous()
                    feature_tmp = features.transpose(1, 2)[:, last_sample_end_index:sample_range, :]
                    cls_features_tmp = cls_features[:, last_sample_end_index:sample_range, :] if cls_features is not None else None 
                    last_sample_end_index += sample_range

                if xyz_tmp.shape[1] <= npoint: # No downsampling
                    sample_idx = torch.arange(xyz_tmp.shape[1], device=xyz_tmp.device, dtype=torch.int32) * torch.ones(xyz_tmp.shape[0], xyz_tmp.shape[1], device=xyz_tmp.device, dtype=torch.int32)

                elif ('cls' in sample_type) or ('ctr' in sample_type):
                    cls_features_max, class_pred = cls_features_tmp.max(dim=-1)
                    score_pred = torch.sigmoid(cls_features_max) # B,N
                    
                    # input_semantic_score = score_pred[0,:].unsqueeze(-1)
                    # print("xyz的尺寸:",xyz.shape,"  语义分数：",input_semantic_score.max())
                    # semantic_view.get_vis(xyz[0,:,:],input_semantic_score)
                    
                    
                    score_picked, sample_idx = torch.topk(score_pred, npoint, dim=-1)           
                    sample_idx = sample_idx.int()

                elif 'D-FPS' in sample_type or 'DFS' in sample_type:
                    sample_idx = pointnet2_utils.furthest_point_sample(xyz_tmp.contiguous(), npoint)

                elif 'F-FPS' in sample_type or 'FFS' in sample_type:
                    features_SSD = torch.cat([xyz_tmp, feature_tmp], dim=-1)
                    features_for_fps_distance = self.calc_square_dist(features_SSD, features_SSD)
                    features_for_fps_distance = features_for_fps_distance.contiguous()
                    sample_idx = pointnet2_utils.furthest_point_sample_with_dist(features_for_fps_distance, npoint)

                elif sample_type == 'FS':
                    features_SSD = torch.cat([xyz_tmp, feature_tmp], dim=-1)
                    features_for_fps_distance = self.calc_square_dist(features_SSD, features_SSD)
                    features_for_fps_distance = features_for_fps_distance.contiguous()
                    sample_idx_1 = pointnet2_utils.furthest_point_sample_with_dist(features_for_fps_distance, npoint)
                    sample_idx_2 = pointnet2_utils.furthest_point_sample(xyz_tmp, npoint)
                    sample_idx = torch.cat([sample_idx_1, sample_idx_2], dim=-1)  # [bs, npoint * 2]
                elif 'Rand' in sample_type:
                    sample_idx = torch.randperm(xyz_tmp.shape[1],device=xyz_tmp.device)[None, :npoint].int().repeat(xyz_tmp.shape[0], 1)
                elif sample_type == 'ds_FPS' or sample_type == 'ds-FPS':
                    part_num = 4
                    xyz_div = []
                    idx_div = []
                    for i in range(len(xyz_tmp)):
                        per_xyz = xyz_tmp[i]
                        radii = per_xyz.norm(dim=-1) -5 
                        storted_radii, indince = radii.sort(dim=0, descending=False)
                        per_xyz_sorted = per_xyz[indince]
                        per_xyz_sorted_div = per_xyz_sorted.view(part_num, -1 ,3)

                        per_idx_div = indince.view(part_num,-1)
                        xyz_div.append(per_xyz_sorted_div)
                        idx_div.append(per_idx_div)
                    xyz_div = torch.cat(xyz_div ,dim=0)
                    idx_div = torch.cat(idx_div ,dim=0)
                    idx_sampled = pointnet2_utils.furthest_point_sample(xyz_div, (npoint//part_num))

                    indince_div = []
                    for idx_sampled_per, idx_per in zip(idx_sampled, idx_div):                    
                        indince_div.append(idx_per[idx_sampled_per.long()])
                    index = torch.cat(indince_div, dim=-1)
                    sample_idx = index.reshape(xyz.shape[0], npoint).int()

                elif sample_type == 'ry_FPS' or sample_type == 'ry-FPS':
                    part_num = 4
                    xyz_div = []
                    idx_div = []
                    for i in range(len(xyz_tmp)):
                        per_xyz = xyz_tmp[i]
                        ry = torch.atan(per_xyz[:,0]/per_xyz[:,1])
                        storted_ry, indince = ry.sort(dim=0, descending=False)
                        per_xyz_sorted = per_xyz[indince]
                        per_xyz_sorted_div = per_xyz_sorted.view(part_num, -1 ,3)

                        per_idx_div = indince.view(part_num,-1)
                        xyz_div.append(per_xyz_sorted_div)
                        idx_div.append(per_idx_div)
                    xyz_div = torch.cat(xyz_div ,dim=0)
                    idx_div = torch.cat(idx_div ,dim=0)
                    idx_sampled = pointnet2_utils.furthest_point_sample(xyz_div, (npoint//part_num))

                    indince_div = []
                    for idx_sampled_per, idx_per in zip(idx_sampled, idx_div):                    
                        indince_div.append(idx_per[idx_sampled_per.long()])
                    index = torch.cat(indince_div, dim=-1)

                    sample_idx = index.reshape(xyz.shape[0], npoint).int()

                sampled_idx_list.append(sample_idx)

            sampled_idx_list = torch.cat(sampled_idx_list, dim=-1) 
            new_xyz = pointnet2_utils.gather_operation(xyz_flipped, sampled_idx_list).transpose(1, 2).contiguous()
            # if len(new_xyz[0,:,:])==1024:
            #     print("new_xyz[0,:,:]",len(new_xyz[0,:,:]))
            #     semantic_view.get_vis_single(new_xyz[0,:,:],torch.ones_like(new_xyz[0,:,:]))
                
        else:
            new_xyz = ctr_xyz

        if len(self.groupers) > 0:
            for i in range(len(self.groupers)):
                new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)
                new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
                if self.pool_method == 'max_pool':
                    new_features = F.max_pool2d(
                        new_features, kernel_size=[1, new_features.size(3)]
                    )  # (B, mlp[-1], npoint, 1)
                elif self.pool_method == 'avg_pool':
                    new_features = F.avg_pool2d(
                        new_features, kernel_size=[1, new_features.size(3)]
                    )  # (B, mlp[-1], npoint, 1)
                else:
                    raise NotImplementedError

                new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
                # 
                new_features_list.append(new_features)

            new_features = torch.cat(new_features_list, dim=1)

            if self.aggregation_layer is not None:
                new_features = self.aggregation_layer(new_features)
        else:
            new_features = pointnet2_utils.gather_operation(features, sampled_idx_list).contiguous()

        if self.confidence_layers is not None:
            cls_features = self.confidence_layers(new_features).transpose(1, 2)            
        else:
            cls_features = None

        return new_xyz, new_features, cls_features,sampled_idx_list


class Vote_layer(nn.Module):
    """ Light voting module with limitation"""
    def __init__(self, mlp_list, pre_channel, max_translate_range):
        super().__init__()
        self.mlp_list = mlp_list
        if len(mlp_list) > 0:
            for i in range(len(mlp_list)):
                shared_mlps = []

                shared_mlps.extend([
                    nn.Conv1d(pre_channel, mlp_list[i], kernel_size=1, bias=False),
                    nn.BatchNorm1d(mlp_list[i]),
                    nn.ReLU()
                ])
                pre_channel = mlp_list[i]
            self.mlp_modules = nn.Sequential(*shared_mlps)
        else:
            self.mlp_modules = None

        self.ctr_reg = nn.Conv1d(pre_channel, 3, kernel_size=1)
        self.max_offset_limit = torch.tensor(max_translate_range).float() if max_translate_range is not None else None
       

    def forward(self, xyz, features):
        xyz_select = xyz
        features_select = features

        if self.mlp_modules is not None: 
            new_features = self.mlp_modules(features_select) #([4, 256, 256]) ->([4, 128, 256])            
        else:
            new_features = new_features
        
        ctr_offsets = self.ctr_reg(new_features) #[4, 128, 256]) -> ([4, 3, 256])

        ctr_offsets = ctr_offsets.transpose(1, 2)#([4, 256, 3])
        feat_offets = ctr_offsets[..., 3:]
        new_features = feat_offets
        ctr_offsets = ctr_offsets[..., :3]
        
        if self.max_offset_limit is not None:
            max_offset_limit = self.max_offset_limit.view(1, 1, 3)            
            max_offset_limit = self.max_offset_limit.repeat((xyz_select.shape[0], xyz_select.shape[1], 1)).to(xyz_select.device) #([4, 256, 3])
      
            limited_ctr_offsets = torch.where(ctr_offsets > max_offset_limit, max_offset_limit, ctr_offsets)
            min_offset_limit = -1 * max_offset_limit
            limited_ctr_offsets = torch.where(limited_ctr_offsets < min_offset_limit, min_offset_limit, limited_ctr_offsets)
            vote_xyz = xyz_select + limited_ctr_offsets
        else:
            # B,N,3
            vote_xyz = xyz_select + ctr_offsets

        # show_vote_pts = vote_xyz[0,:,:]  # 展示的中心点  N,3
        # show_vote_score =  torch.ones_like(show_vote_pts)[:,:1]  # N,1
        # show_xyz_select = xyz_select[0,:,:]
        # show_xyz_score =  torch.ones_like(show_xyz_select)[:,:1]//2

        # show_vote_pts  = torch.cat([show_vote_pts,show_xyz_select ],dim=0)
        # show_vote_score  = torch.cat([show_vote_score,show_xyz_score ],dim=0)

        # semantic_view.get_vis(show_vote_pts,show_vote_score)
                    
        # # semantic_view.get_vis(new_xyz[0,:,:],torch.ones_like(new_xyz)[0,:,:1])

        # vote_xyz 偏移之后的中心点、偏移之后的特征（一般为空）、原始（偏移）之前的点、偏移量
        return vote_xyz, new_features, xyz_select, ctr_offsets


class PointnetSAModule(PointnetSAModuleMSG):
    """Pointnet set abstraction layer"""

    def __init__(self, *, mlp: List[int], npoint: int = None, radius: float = None, nsample: int = None,
                 bn: bool = True, use_xyz: bool = True, pool_method='max_pool'):
        """
        :param mlp: list of int, spec of the pointnet before the global max_pool
        :param npoint: int, number of features
        :param radius: float, radius of ball
        :param nsample: int, number of samples in the ball query
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        """
        super().__init__(
            mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample], bn=bn, use_xyz=use_xyz,
            pool_method=pool_method
        )


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another"""

    def __init__(self, *, mlp: List[int], bn: bool = True):
        """
        :param mlp: list of int
        :param bn: whether to use batchnorm
        """
        super().__init__()

        shared_mlps = []
        for k in range(len(mlp) - 1):
            shared_mlps.extend([
                nn.Conv2d(mlp[k], mlp[k + 1], kernel_size=1, bias=False),
                nn.BatchNorm2d(mlp[k + 1]),
                nn.ReLU()
            ])
        self.mlp = nn.Sequential(*shared_mlps)

    def forward(
            self, unknown: torch.Tensor, known: torch.Tensor, unknow_feats: torch.Tensor, known_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        :param unknown: (B, n, 3) tensor of the xyz positions of the unknown features
        :param known: (B, m, 3) tensor of the xyz positions of the known features
        :param unknow_feats: (B, C1, n) tensor of the features to be propigated to
        :param known_feats: (B, C2, m) tensor of features to be propigated
        :return:
            new_features: (B, mlp[-1], n) tensor of the features of the unknown features
        """
        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))

        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)


if __name__ == "__main__":
    pass
