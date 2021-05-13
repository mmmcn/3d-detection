#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include <vector>

extern THCState *state;


// b: batch_size
// num_voxel: 体素块个数,填充过的
// num_points: 体素块内点个数,填充过的
// m: 采样点个数
// 不确定是否内存会溢出 [4, 750, 1500, 3] 4个G?
// fp16估计会好点

int voxel_furthest_point_sampling_wrapper(int b, int n, int m,
                                          at::Tensor points_xyz,
                                          at::Tensor weights_xyz,
                                          at::Tensor temp_tensor,
                                          at::Tensor idx_tensor);

void voxel_furthest_point_sampling_kernel_launcher(int b, int n, int m,
                                                   const float *dataset,
                                                   const float *weights,
                                                   float *temp, int *idxs,
                                                   cudaStream_t stream);

int voxel_furthest_point_sampling_wrapper(int b, int n, int m,
                                          at::Tensor points_xyz,
                                          at::Tensor weights_xyz,
                                          at::Tensor temp_tensor,
                                          at::Tensor idx_tensor) {

    const float *points = points_xyz.data_ptr<float>();
    const float *weights = weights_xyz.data_ptr<float>();
    float *temp = temp_tensor.data_ptr<float>();
    int *idx = idx_tensor.data_ptr<int>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    voxel_furthest_point_sampling_kernel_launcher(b, n, m, points, weights, temp, idx, stream);
    return 1;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("voxel_furthest_point_sampling_wrapper", &voxel_furthest_point_sampling_wrapper,
        "voxel_furthest_point_sampling_wrapper");
}