#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include <vector>

extern THCState *state;


// weights: [batch_size, n]
int density_weights_furthest_point_sampling_wrapper(int b, int n, int m,
                                                    at::Tensor points_tensor,
                                                    at::Tensor weights,
                                                    at::Tensor temp_tensor,
                                                    at::Tensor idx_tensor);

void density_weights_furthest_point_sampling_kernel_launcher(int b, int n, int m,
                                                             const float *dataset,
                                                             const float *weights,
                                                             float *temp, int *idxs,
                                                             cudaStream_t stream);

int furthest_point_sampling_with_dist_wrapper(int b, int n, int m,
                                              at::Tensor points_tensor,
                                              at::Tensor temp_tensor,
                                              at::Tensor idx_tensor);

void furthest_point_sampling_with_dist_kernel_launcher(int b, int n, int m,
                                                       const float *dataset,
                                                       float *temp, int *idxs,
                                                       cudaStream_t stream);


int num_points_within_r_wrapper(int b, int n, float r,
                                 at::Tensor points_tensor,
                                 at::Tensor results_tensor);

void num_points_within_r_kernel_launcher(int b, int n, float r,
                                         const float *dataset,
                                         float *results,
                                         cudaStream_t stream);


int num_points_within_r_wrapper(int b, int n, float r,
                                at::Tensor points_tensor,
                                at::Tensor result_tensor) {
    const float *points = points_tensor.data_ptr<float>();
    float *results = result_tensor.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    num_points_within_r_kernel_launcher(b, n, r, points, results, stream);
    return 1;

}


int density_weights_furthest_point_sampling_wrapper(int b, int n, int m,
                                                    at::Tensor points_tensor,
                                                    at::Tensor weights_tensor,
                                                    at::Tensor temp_tensor,
                                                    at::Tensor idx_tensor) {

    const float *points = points_tensor.data_ptr<float>();
    const float *weights = weights_tensor.data_ptr<float>();

    float *temp = temp_tensor.data_ptr<float>();
    int *idx = idx_tensor.data_ptr<int>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    density_weights_furthest_point_sampling_kernel_launcher(b, n, m,
                                                            points, weights,
                                                            temp, idx, stream);
    return 1;
}

int furthest_point_sampling_with_dist_wrapper(int b, int n, int m,
                                              at::Tensor points_tensor,
                                              at::Tensor temp_tensor,
                                              at::Tensor idx_tensor) {

  const float *points = points_tensor.data<float>();
  float *temp = temp_tensor.data<float>();
  int *idx = idx_tensor.data<int>();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  furthest_point_sampling_with_dist_kernel_launcher(b, n, m, points, temp, idx, stream);
  return 1;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("num_points_within_r_wrapper", &num_points_within_r_wrapper,
        "num_points_within_r_wrapper");
  m.def("density_weights_furthest_point_sampling_wrapper",
        &density_weights_furthest_point_sampling_wrapper,
        "density_weights_furthest_point_sampling_wrapper");
   m.def("furthest_point_sampling_with_dist_wrapper",
        &furthest_point_sampling_with_dist_wrapper,
        "furthest_point_sampling_with_dist_wrapper");
}