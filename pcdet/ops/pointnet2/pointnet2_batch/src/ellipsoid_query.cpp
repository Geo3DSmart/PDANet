// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "ellipsoid_query_gpu.h"
#include "utils.h"

void query_ellipsoid_point_kernel_wrapper(int b, int n, int m, float e1, float e2, float e3,
                                     int nsample, const float *new_xyz,
                                     const float *xyz, int *idx, int *ingroup_pts_cnt, float *ingroup_out, float *ingroup_cva, float *v, float *d);

at::Tensor ellipsoid_query(at::Tensor new_xyz, at::Tensor xyz, const float e1, const float e2, const float e3,
                      const int nsample) {
  CHECK_CONTIGUOUS(new_xyz);
  CHECK_CONTIGUOUS(xyz);
  // CHECK_CONTIGUOUS()
  CHECK_IS_FLOAT(new_xyz);
  CHECK_IS_FLOAT(xyz);

  if (new_xyz.type().is_cuda()) {
    CHECK_CUDA(xyz);
  }

  at::Tensor idx =
      torch::zeros({new_xyz.size(0), new_xyz.size(1), nsample},
                   at::device(new_xyz.device()).dtype(at::ScalarType::Int));

 at::Tensor ingroup_pts_cnt =
     torch::zeros({new_xyz.size(0), new_xyz.size(1)},
                  at::device(new_xyz.device()).dtype(at::ScalarType::Int));

at::Tensor ingroup_out =
    torch::zeros({new_xyz.size(0), new_xyz.size(1), nsample, 3},
                 at::device(new_xyz.device()).dtype(at::ScalarType::Float));

  at::Tensor ingroup_cva =
       torch::zeros({new_xyz.size(0), new_xyz.size(1), 9},
                    at::device(new_xyz.device()).dtype(at::ScalarType::Float));

  at::Tensor v =
      torch::zeros({new_xyz.size(0), new_xyz.size(1), 9},
                   at::device(new_xyz.device()).dtype(at::ScalarType::Float));
 at::Tensor d =
     torch::zeros({new_xyz.size(0), new_xyz.size(1), 3},
                  at::device(new_xyz.device()).dtype(at::ScalarType::Float));

  // at::Tensor distance =
  //     torch::zeros({new_xyz.size(0), new_xyz.size(1), nsample},
  //                  at::device(new_xyz.device()).dtype(at::ScalarType::Float));

  if (new_xyz.type().is_cuda()) {
    query_ellipsoid_point_kernel_wrapper(xyz.size(0), xyz.size(1), new_xyz.size(1),
                                    e1, e2, e3, nsample, new_xyz.data<float>(),
                                    xyz.data<float>(), idx.data<int>(), ingroup_pts_cnt.data<int>(), ingroup_out.data<float>(), ingroup_cva.data<float>(), v.data<float>(), d.data<float>());
  } else {
    AT_CHECK(false, "CPU not supported");
  }
  //
  // struct idxd{
  //   at::Tensor ridx;
  //   at::Tensor d;
  // };
  // typedef struct idxd Struct;
  // Struct id;
  //
  // id.ridx = idx;
  // id.d = distance;
  // std::vector<torch::Tensor> outputs;
  // auto out = torch::zeros({1,2}, torch::dtype(torch::kFloat32));
  // for (int n=0; n<N; n++)
  // outputs.push_back(idx);
  // outputs.push_back(distance);

  return idx;
}


//
// int ellipsoid_query_wrapper(int b, int n, int m, float e1, float e2, float e3, int nsample,
// 		       THCudaTensor *new_xyz_tensor, THCudaTensor *xyz_tensor, THCudaIntTensor *fps_idx_tensor,
// 		       THCudaIntTensor *idx_tensor,THCudaIntTensor  *ingroup_pts_cnt_tensor, THCudaTensor *ingroup_out_tensor, THCudaTensor *ingroup_cva_tensor, THCudaTensor *v_tensor,THCudaTensor *d_tensor) {
//
//     const float *new_xyz = THCudaTensor_data(state, new_xyz_tensor);
//     const float *xyz = THCudaTensor_data(state, xyz_tensor);
//     const int *fps_idx = THCudaIntTensor_data(state, fps_idx_tensor);
//     int *idx = THCudaIntTensor_data(state, idx_tensor);
//     //below tensors added by me
//     int *ingroup_pts_cnt = THCudaIntTensor_data(state, ingroup_pts_cnt_tensor);
//     float *ingroup_out = THCudaTensor_data(state, ingroup_out_tensor);
//     float *ingroup_cva = THCudaTensor_data(state, ingroup_cva_tensor);
//     float *v = THCudaTensor_data(state, v_tensor);
//     float *d = THCudaTensor_data(state, d_tensor);
//
//     cudaStream_t stream = THCState_getCurrentStream(state);
//
//     query_ellipsoid_point_kernel_wrapper(b, n, m, e1, e2, e3, nsample, new_xyz, xyz, fps_idx, idx, ingroup_pts_cnt, ingroup_out, ingroup_cva, v, d,
// 				    stream);
//     return 1;
// }
