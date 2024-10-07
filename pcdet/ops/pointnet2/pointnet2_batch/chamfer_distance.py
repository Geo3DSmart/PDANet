import importlib
import os

import torch
from torch import nn
from torch.autograd import Function

from . import pointnet2_batch_cuda as pointnet2


# chamfer_found = importlib.find_loader("chamfer_3D") is not None
# if not chamfer_found:
#     ## Cool trick from https://github.com/chrdiller
#     print("Jitting Chamfer 3D")

#     from torch.utils.cpp_extension import load
#     chamfer_3D = load(name="chamfer_3D",
#           sources=[
#               "/".join(os.path.abspath(__file__).split('/')[:-1] + ["chamfer_cuda.cpp"]),
#               "/".join(os.path.abspath(__file__).split('/')[:-1] + ["chamfer3D.cu"]),
#               ])
#     # print("Loaded JIT 3D CUDA chamfer distance")

# else:
#     import chamfer_3D
#     # print("Loaded compiled 3D CUDA chamfer distance")


# Chamfer's distance module @thibaultgroueix
# GPU tensors only
class chamfer_3DFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        """
        xyz1: (B, N, 3)
        xyz2: (B, M, 3)
        """
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        device = xyz1.device

        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)

        dist1 = dist1.to(device)
        dist2 = dist2.to(device)
        idx1 = idx1.to(device)
        idx2 = idx2.to(device)
        torch.cuda.set_device(device)

        pointnet2.chamfer_forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, graddist1, graddist2, gradidx1, gradidx2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()
        device = graddist1.device

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        gradxyz1 = gradxyz1.to(device)
        gradxyz2 = gradxyz2.to(device)
        pointnet2.chamfer_backward(
            xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2
        )
        return gradxyz1, gradxyz2


class ChamferDistance(nn.Module):
    def __init__(self):
        super(ChamferDistance, self).__init__()

    def forward(self, input1, input2):
        """
        input1: (B, N, 3)
        input2: (B, M, 3)
        """
        dist1, dist2, _, _ = chamfer_3DFunction.apply(input1, input2)
        return dist1, dist2  