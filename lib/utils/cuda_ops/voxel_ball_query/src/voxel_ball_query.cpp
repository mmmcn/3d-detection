// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/ball_query.cpp

#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include <vector>

extern THCState *state;

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)


int voxel_ball_query_wrapper(int b, int n, int m, float min_radius, float max_radius, int nsample,
                             at::Tensor new_xyz_tensor, at::Tensor xyz_tensor,
                             at::Tensor idx_tensor);

void voxel_ball_query_kernel_launcher(int b, int n, int m, float min_radius, float max_radius,
                                      int nsample, const float *new_xyz, const float *xyz,
                                      int *idx, cudaStream_t stream);

int voxel_ball_query_wrapper(int b, int n, int m, float min_radius, float max_radius, int nsample,
                             at::Tensor new_xyz_tensor, at::Tensor xyz_tensor,
                             at::Tensor idx_tensor) {
  CHECK_INPUT(new_xyz_tensor);
  CHECK_INPUT(xyz_tensor);
  const float *new_xyz = new_xyz_tensor.data_ptr<float>();
  const float *xyz = xyz_tensor.data_ptr<float>();
  int *idx = idx_tensor.data_ptr<int>();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  voxel_ball_query_kernel_launcher(b, n, m, min_radius, max_radius,
                                   nsample, new_xyz, xyz, idx, stream);
  return 1;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("voxel_ball_query_wrapper", &voxel_ball_query_wrapper, "voxel_ball_query_wrapper");
}