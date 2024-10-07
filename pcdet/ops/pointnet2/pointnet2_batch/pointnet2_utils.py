from typing import Tuple

import torch
import torch.nn as nn
from torch.autograd import Function, Variable

from . import pointnet2_batch_cuda as pointnet2


class FarthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """
        Uses iterative farthest point sampling to select a set of npoint features that have the largest
        minimum distance
        :param ctx:
        :param xyz: (B, N, 3) where N > npoint
        :param npoint: int, number of features in the sampled set
        :return:
             output: (B, npoint) tensor containing the set
        """
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        pointnet2.farthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


farthest_point_sample = furthest_point_sample = FarthestPointSampling.apply


class FurthestPointSamplingWithDist(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance
        :param ctx:
        :param xyz: (B, N, N) where N > npoint
        :param npoint: int, number of features in the sampled set
        :return:
             output: (B, npoint) tensor containing the set
        """
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        return_int = pointnet2.furthest_point_sampling_with_dist_wrapper(B, N, npoint, xyz, temp, output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample_with_dist = FurthestPointSamplingWithDist.apply

class GatherOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param features: (B, C, N)
        :param idx: (B, npoint) index tensor of the features to gather
        :return:
            output: (B, C, npoint)
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, npoint = idx.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, npoint)

        pointnet2.gather_points_wrapper(B, C, N, npoint, features, idx, output)

        ctx.for_backwards = (idx, C, N)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards
        B, npoint = idx.size()

        grad_features = Variable(torch.cuda.FloatTensor(B, C, N).zero_())
        grad_out_data = grad_out.data.contiguous()
        pointnet2.gather_points_grad_wrapper(B, C, N, npoint, grad_out_data, idx, grad_features.data)
        return grad_features, None


gather_operation = GatherOperation.apply


class ThreeNN(Function):

    @staticmethod
    def forward(ctx, unknown: torch.Tensor, known: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the three nearest neighbors of unknown in known
        :param ctx:
        :param unknown: (B, N, 3)
        :param known: (B, M, 3)
        :return:
            dist: (B, N, 3) l2 distance to the three nearest neighbors
            idx: (B, N, 3) index of 3 nearest neighbors
        """
        assert unknown.is_contiguous()
        assert known.is_contiguous()

        B, N, _ = unknown.size()
        m = known.size(1)
        dist2 = torch.cuda.FloatTensor(B, N, 3)
        idx = torch.cuda.IntTensor(B, N, 3)

        pointnet2.three_nn_wrapper(B, N, m, unknown, known, dist2, idx)
        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Performs weight linear interpolation on 3 features
        :param ctx:
        :param features: (B, C, M) Features descriptors to be interpolated from
        :param idx: (B, n, 3) three nearest neighbors of the target features in features
        :param weight: (B, n, 3) weights
        :return:
            output: (B, C, N) tensor of the interpolated features
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()
        assert weight.is_contiguous()

        B, c, m = features.size()
        n = idx.size(1)
        ctx.three_interpolate_for_backward = (idx, weight, m)
        output = torch.cuda.FloatTensor(B, c, n)

        pointnet2.three_interpolate_wrapper(B, c, m, n, features, idx, weight, output)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param ctx:
        :param grad_out: (B, C, N) tensor with gradients of outputs
        :return:
            grad_features: (B, C, M) tensor with gradients of features
            None:
            None:
        """
        idx, weight, m = ctx.three_interpolate_for_backward
        B, c, n = grad_out.size()

        grad_features = Variable(torch.cuda.FloatTensor(B, c, m).zero_())
        grad_out_data = grad_out.data.contiguous()

        pointnet2.three_interpolate_grad_wrapper(B, c, n, m, grad_out_data, idx, weight, grad_features.data)
        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply


class GroupingOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param features: (B, C, N) tensor of features to group
        :param idx: (B, npoint, nsample) tensor containing the indicies of features to group with
        :return:
            output: (B, C, npoint, nsample) tensor
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, nfeatures, nsample)

        pointnet2.group_points_wrapper(B, C, N, nfeatures, nsample, features, idx, output)

        ctx.for_backwards = (idx, N)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param ctx:
        :param grad_out: (B, C, npoint, nsample) tensor of the gradients of the output from forward
        :return:
            grad_features: (B, C, N) gradient of the features
        """
        idx, N = ctx.for_backwards

        B, C, npoint, nsample = grad_out.size()
        grad_features = Variable(torch.cuda.FloatTensor(B, C, N).zero_())

        grad_out_data = grad_out.data.contiguous()
        pointnet2.group_points_grad_wrapper(B, C, N, npoint, nsample, grad_out_data, idx, grad_features.data)
        return grad_features, None


grouping_operation = GroupingOperation.apply


class BallQuery(Function):

    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param radius: float, radius of the balls
        :param nsample: int, maximum number of features in the balls
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centers of the ball query
        :return:
            idx: (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        assert new_xyz.is_contiguous()
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        npoint = new_xyz.size(1)
        idx = torch.cuda.IntTensor(B, npoint, nsample).zero_()

        pointnet2.ball_query_wrapper(B, N, npoint, radius, nsample, new_xyz, xyz, idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply

class BallQueryDilated(Function):

    @staticmethod
    def forward(ctx, max_radius: float, min_radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param max_radius: float, max radius of the balls
        :param min_radius: float, min radius of the balls
        :param nsample: int, maximum number of features in the balls
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centers of the ball query
        :return:
            idx: (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        assert new_xyz.is_contiguous()
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        npoint = new_xyz.size(1)
        idx = torch.cuda.IntTensor(B, npoint, nsample).zero_()

        pointnet2.ball_query_dilated_wrapper(B, N, npoint, max_radius, min_radius, nsample, new_xyz, xyz, idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None


ball_query_dilated = BallQueryDilated.apply




class EllipsoidQuery(Function):
    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz):
        # type: (Any, float, int, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        radius : float
            radius of the balls
        nsample : int
            maximum number of features in the balls
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the ball query

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        inds = pointnet2.ellipsoid_query(new_xyz, xyz, radius, radius*2, radius, nsample)
        # inds = _ext.ball_query(new_xyz, xyz, radius, nsample)
        ctx.mark_non_differentiable(inds)
        return inds

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ellipsoid_query = EllipsoidQuery.apply




class QueryAndGroup_Ellipsoid(nn.Module):
    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        """
        :param radius: float, radius of ball
        :param nsample: int, maximum number of features to gather in the ball
        :param use_xyz:
        """
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        # idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        idx = ellipsoid_query(self.radius, self.nsample, xyz, new_xyz)
        
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features

def compute_density(new_xyz_k,center_xyz, bandwidth):
    '''
        new_xyz_k:[B,N,k,3]
        xyz: [B,N,3]
    return [B,N,K]
    '''
    #import ipdb; ipdb.set_trace()
    # unsqueeze 函数来将中心点的坐标张量从 (B,N,3) 扩展成 (B,N,1,3)
    distances = torch.norm(new_xyz_k - center_xyz.unsqueeze(2), dim=-1) # dist:(B,N,K)
    
    gaussion_density = torch.exp(-distances**2 / (2*bandwidth**2))/(2.5*bandwidth)
    
    # 这个进行激活
    gaussion_density  = gaussion_density.mean(dim = -1)  # B，N
    
    # B, N, C = xyz.shape
    # sqrdists = square_distance(xyz, origin_xyz) 
    # gaussion_density = torch.exp(- sqrdists / (2.0 * bandwidth * bandwidth)) / (2.5 * bandwidth)
    # xyz_density = gaussion_density.mean(dim = -1)

    return gaussion_density 

def compute_density_Ellipsoid(new_xyz_k,center_xyz, bandwidth):
    '''
        new_xyz_k:[B,N,k,3]
        xyz: [B,N,3]
    return [B,N,K]
    '''
    #import ipdb; ipdb.set_trace()
    # unsqueeze 函数来将中心点的坐标张量从 (B,N,3) 扩展成 (B,N,1,3)
    distances = torch.norm(new_xyz_k - center_xyz.unsqueeze(2), dim=-1) # dist:(B,N,K)
    
    gaussion_density = torch.exp(-distances**2 / (2*bandwidth**2))/(2.5*bandwidth)
    
    # 这个进行激活
    # gaussion_density  = gaussion_density.mean(dim = -1)  # B，N
    
    # B, N, C = xyz.shape
    # sqrdists = square_distance(xyz, origin_xyz) 
    # gaussion_density = torch.exp(- sqrdists / (2.0 * bandwidth * bandwidth)) / (2.5 * bandwidth)
    # xyz_density = gaussion_density.mean(dim = -1)

    return gaussion_density 



class QueryAndGroup_grouped_density_Ellipsoid(nn.Module):
    def __init__(self, radius: float, nsample: int,bandwidth:int,use_xyz: bool = True):
        """
        :param radius: float, radius of ball
        :param nsample: int, maximum number of features to gather in the ball
        :param use_xyz:
        """
        super().__init__()
        self.radius, self.nsample,self.bandwidth, self.use_xyz = radius, nsample,bandwidth, use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        # 基于密度的
        # idx_density = ball_query(self.radius, self.nsample, xyz, xyz) # idx_density: (B,npoint, nsample)
        idx = ellipsoid_query(self.radius, self.nsample, xyz, new_xyz)
  
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
  
        grouped_xyz_density = grouped_xyz.permute(0, 2,3, 1).contiguous() #  grouped_xyz_density:(B, 3, npoint, nsample)->(B, npoint, nsample, 3)
        xyz_density = compute_density_Ellipsoid(grouped_xyz_density,new_xyz,self.bandwidth)  #  xyz_density:B， npoint, nsample
        inverse_density = 1.0 / xyz_density 
        grouped_density = inverse_density.unsqueeze(1) # [B,1,npoint, nsample]
       
       
        
        # idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        # xyz_trans = xyz.transpose(1, 2).contiguous()
        # grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        
        # grouped_density_feature = grouping_operation(grouped_density,idx) # (B, 1, npoint, nsample)
        
        # grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz,grouped_density, grouped_features], dim=1)  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features





class QueryAndGroup_grouped_density(nn.Module):
    def __init__(self, radius: float, nsample: int,bandwidth:int,use_xyz: bool = True):
        """
        :param radius: float, radius of ball
        :param nsample: int, maximum number of features to gather in the ball
        :param use_xyz:
        """
        super().__init__()
        self.radius, self.nsample,self.bandwidth, self.use_xyz = radius, nsample,bandwidth, use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        # 基于密度的
        idx_density = ball_query(self.radius, self.nsample, xyz, xyz) # idx_density: (B, N, nsample)
        # grouped_xyz_density: # (B, 3, N, nsample)
        grouped_xyz_density = grouping_operation(xyz.transpose(1, 2).contiguous(), idx_density)  # (B,3, N, nsample)
        grouped_xyz_density = grouped_xyz_density.permute(0, 2,3, 1).contiguous() #  grouped_xyz_density:(B, 3, N, nsample)->(B, N, nsample, 3)
        xyz_density = compute_density(grouped_xyz_density,xyz,self.bandwidth)  #  xyz_density:B，N
        inverse_density = 1.0 / xyz_density 
        grouped_density = inverse_density.unsqueeze(1) # [B,1,N]
       
       
        
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        
        grouped_density_feature = grouping_operation(grouped_density,idx) # (B, 1, npoint, nsample)
        
        # grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz,grouped_density_feature, grouped_features], dim=1)  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features




class QueryAndGroup_grouped_xyz(nn.Module):
    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        """
        :param radius: float, radius of ball
        :param nsample: int, maximum number of features to gather in the ball
        :param use_xyz:
        """
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        
        # grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features


class QueryAndGroup_alone_grouped_density_directional(nn.Module):
    def __init__(self, radius: float, nsample: int,use_xyz: bool = True):
        """
        :param radius: float, radius of ball
        :param nsample: int, maximum number of features to gather in the ball
        :param use_xyz:
        """
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        # 基于密度的  这个也可以改
        # idx_density = ball_query(self.radius, self.nsample, xyz, xyz) # idx_density: (B, N, nsample)
        # # grouped_xyz_density: # (B, 3, N, nsample)
        # grouped_xyz_density = grouping_operation(xyz.transpose(1, 2).contiguous(), idx_density)  # (B,3, N, nsample)
        # grouped_xyz_density = grouped_xyz_density.permute(0, 2,3, 1).contiguous() #  grouped_xyz_density:(B, 3, N, nsample)->(B, N, nsample, 3)
        # xyz_density = compute_density(grouped_xyz_density,xyz,self.radius)  #  xyz_density:B，N
        # inverse_density = 1.0 / xyz_density 
        # grouped_density = inverse_density.unsqueeze(1) # [B,1,N]
       
       
        
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        # idx = ellipsoid_query(self.radius, self.nsample, xyz, new_xyz)
        
        
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        
        
        distances = torch.norm(grouped_xyz.permute(0, 2,3, 1).contiguous() - new_xyz.unsqueeze(2), dim=-1) # dist:(B,N,K)
        gaussion_density = torch.exp(-distances**2 / (2*self.radius**2))/(2.5*self.radius)# B,npoint,k 
        # gaussion_density = 1.0 / gaussion_density
        gaussion_density = gaussion_density.unsqueeze(-1).permute(0, 3,1,2).contiguous()
        
        directional_vectors = grouped_xyz-new_xyz.transpose(1, 2).unsqueeze(-1)  # # (B, 3, npoint, nsample)
        directional_vectors  = directional_vectors / self.radius
        
        # grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz,gaussion_density,directional_vectors, grouped_features], dim=1)  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features



class QueryAndGroup_alone_grouped_density(nn.Module):
    def __init__(self, radius: float, nsample: int,use_xyz: bool = True):
        """
        :param radius: float, radius of ball
        :param nsample: int, maximum number of features to gather in the ball
        :param use_xyz:
        """
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        # 基于密度的  这个也可以改
        # idx_density = ball_query(self.radius, self.nsample, xyz, xyz) # idx_density: (B, N, nsample)
        # # grouped_xyz_density: # (B, 3, N, nsample)
        # grouped_xyz_density = grouping_operation(xyz.transpose(1, 2).contiguous(), idx_density)  # (B,3, N, nsample)
        # grouped_xyz_density = grouped_xyz_density.permute(0, 2,3, 1).contiguous() #  grouped_xyz_density:(B, 3, N, nsample)->(B, N, nsample, 3)
        # xyz_density = compute_density(grouped_xyz_density,xyz,self.radius)  #  xyz_density:B，N
        # inverse_density = 1.0 / xyz_density 
        # grouped_density = inverse_density.unsqueeze(1) # [B,1,N]
       
       
        
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        
        
        distances = torch.norm(grouped_xyz.permute(0, 2,3, 1).contiguous() - new_xyz.unsqueeze(2), dim=-1) # dist:(B,N,K)
        gaussion_density = torch.exp(-distances**2 / (2*self.radius**2))/(2.5*self.radius)# B,npoint,k 
        gaussion_density = gaussion_density.unsqueeze(-1).permute(0, 3,1,2).contiguous()
        
        # grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz,gaussion_density, grouped_features], dim=1)  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features


class QueryAndGroup(nn.Module):
    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        """
        :param radius: float, radius of ball
        :param nsample: int, maximum number of features to gather in the ball
        :param use_xyz:
        """
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features

class QueryDilatedAndGroup(nn.Module):
    def __init__(self, radius_in: float, radius_out: float, nsample: int, use_xyz: bool = True):
        """
        :param radius_in: float, radius of inner ball
        :param radius_out: float, radius of outer ball
        :param nsample: int, maximum number of features to gather in the ball
        :param use_xyz:
        """
        super().__init__()
        self.radius_in, self.radius_out, self.nsample, self.use_xyz = radius_in, radius_out, nsample, use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
            idx_cnt: (B, npoint) tensor with the number of grouped points for each ball query
        """
        idx = ball_query_dilated(self.radius_in, self.radius_out, self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features
        
class GroupAll(nn.Module):
    def __init__(self, use_xyz: bool = True):
        super().__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: ignored
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, C + 3, 1, N)
        """
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features
