#ifndef _CHAMFERTHREED_H
#define _CHAMFERTHREED_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>





int chamfer_cuda_forward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor dist1, at::Tensor dist2, at::Tensor idx1, at::Tensor idx2);


int chamfer_cuda_backward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor gradxyz1, at::Tensor gradxyz2, at::Tensor graddist1, at::Tensor graddist2, at::Tensor idx1, at::Tensor idx2);




int chamfer_forward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor dist1, at::Tensor dist2, at::Tensor idx1, at::Tensor idx2) ;


int chamfer_backward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor gradxyz1, at::Tensor gradxyz2, at::Tensor graddist1, 
					  at::Tensor graddist2, at::Tensor idx1, at::Tensor idx2);


#endif