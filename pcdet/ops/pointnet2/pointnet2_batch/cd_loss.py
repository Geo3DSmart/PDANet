import torch

# from extensions.chamfer_distance.chamfer_distance import ChamferDistance
# from extensions.earth_movers_distance.emd import EarthMoverDistance


from . import chamfer_distance 


CD = chamfer_distance.ChamferDistance()
# EMD = EarthMoverDistance()


def cd_loss_L1(pcs1, pcs2):
    """
    L1 Chamfer Distance.

    Args:
        pcs1 (torch.tensor): (B, N, 3)
        pcs2 (torch.tensor): (B, M, 3)
    """
    dist1, dist2 = CD(pcs1, pcs2)
    dist1 = torch.sqrt(dist1)
    # dist2 = torch.sqrt(dist2) # choice
    return (torch.mean(dist1) + torch.mean(dist2)) / 2.0
    # dist1, _ = CD(pcs1, pcs2)
    # dist1 = torch.sqrt(dist1)
    # return torch.mean(dist1)


def cd_loss_L2(pcs1, pcs2):
    """
    L2 Chamfer Distance.

    Args:
        pcs1 (torch.tensor): (B, N, 3)
        pcs2 (torch.tensor): (B, M, 3)
    """
    dist1, dist2 = CD(pcs1, pcs2)
    return torch.mean(dist1) + torch.mean(dist2)
    # _, dist2 = CD(pcs1, pcs2)
    # return  torch.mean(dist2)
    # dist1,_ = CD(pcs1, pcs2)
    # return torch.mean(dist1)


# def emd_loss(pcs1, pcs2):
#     """
#     EMD Loss.

#     Args:
#         xyz1 (torch.Tensor): (b, N, 3)
#         xyz2 (torch.Tensor): (b, N, 3)
#     """
#     dists = EMD(pcs1, pcs2)
#     return torch.mean(dists)
