#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "ball_query_gpu.h"
#include "group_points_gpu.h"
#include "sampling_gpu.h"
#include "interpolate_gpu.h"

#include "ellipsoid_query_gpu.h"
#include "chamferthreed.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ball_query_wrapper", &ball_query_wrapper_fast, "ball_query_wrapper_fast");
    m.def("ball_query_dilated_wrapper", &ball_query_dilated_wrapper_fast, "ball_query_dilated_wrapper_fast");
    
    m.def("ellipsoid_query", &ellipsoid_query, "ellipsoid_query");

    m.def("chamfer_forward", &chamfer_forward, "chamfer forward");
    m.def("chamfer_backward", &chamfer_backward, "chamfer backward");

    m.def("group_points_wrapper", &group_points_wrapper_fast, "group_points_wrapper_fast");
    m.def("group_points_grad_wrapper", &group_points_grad_wrapper_fast, "group_points_grad_wrapper_fast");

    m.def("gather_points_wrapper", &gather_points_wrapper_fast, "gather_points_wrapper_fast");
    m.def("gather_points_grad_wrapper", &gather_points_grad_wrapper_fast, "gather_points_grad_wrapper_fast");

    m.def("farthest_point_sampling_wrapper", &farthest_point_sampling_wrapper, "farthest_point_sampling_wrapper");
    m.def("furthest_point_sampling_with_dist_wrapper", &furthest_point_sampling_with_dist_wrapper, "furthest_point_sampling_with_dist_wrapper");
    
    m.def("three_nn_wrapper", &three_nn_wrapper_fast, "three_nn_wrapper_fast");
    m.def("three_interpolate_wrapper", &three_interpolate_wrapper_fast, "three_interpolate_wrapper_fast");
    m.def("three_interpolate_grad_wrapper", &three_interpolate_grad_wrapper_fast, "three_interpolate_grad_wrapper_fast");
}
