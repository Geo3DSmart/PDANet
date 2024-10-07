// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <torch/extension.h>

at::Tensor ellipsoid_query(at::Tensor new_xyz, at::Tensor xyz, const float e1, const float e2, const float e3,
                      const int nsample);
