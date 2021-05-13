// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/ball_query_gpu.cu

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define TOTAL_THREADS 2048
#define THREADS_PER_BLOCK 32
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
#define SM_NUM 80


inline int opt_n_threads(int b) {
  const int max_threads_per_block = (SM_NUM * TOTAL_THREADS) / b;
  const int pow_2 = std::log(static_cast<double>(max_threads_per_block)) / std::log(2.0);

  return max(min(1 << pow_2, TOTAL_THREADS), 1);

}


__global__ void voxel_ball_query_kernel(int b, int n, int m,
                                        float min_radius,
                                        float max_radius,
                                        int nsample,
                                        int block_size,
                                        const float *__restrict__ new_xyz,
                                        const float *__restrict__ xyz,
                                        int *__restrict__ idx) {
  // new_xyz: (B, M, 3)
  // xyz: (B, N, 3)
  // output:
  //      idx: (B, M, nsample)
  int bs_idx = blockIdx.y;
  int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (bs_idx >= b || pt_idx >= m) return;

  const int stride = block_size;
  //printf("bs_idx: %d pt_idx: %d\n", bs_idx, pt_idx);

  xyz += bs_idx * n * 3;
  new_xyz += bs_idx * m * 3;
  idx += bs_idx * m * nsample;

  // 线程总个数可能小于采样点个数.
  for (int k = pt_idx; k < m; k += stride){
    // 问题在这里
    // 指针越界.
    //new_xyz += bs_idx * m * 3 + k * 3;
    //idx += bs_idx * m * nsample + k * nsample;
    //printf("Current k: %d\n", k);

    float max_radius2 = max_radius * max_radius;
    float min_radius2 = min_radius * min_radius;
    // 当前线程处理的采样点坐标
    float new_x = new_xyz[k * 3 + 0];
    float new_y = new_xyz[k * 3 + 1];
    float new_z = new_xyz[k * 3 + 2];

    int cnt = 0;
    for (int j = 0; j < n; ++j) {
        float x = xyz[j * 3 + 0];
        float y = xyz[j * 3 + 1];
        float z = xyz[j * 3 + 2];

        float d2 = 0.0;
        if (x == 0 && y == 0 && z == 0) {
            d2 = max_radius2 + 1.0;
        }
        else {
            d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
             (new_z - z) * (new_z - z);
        }

        if (d2 == 0 || (d2 >= min_radius2 && d2 < max_radius2)) {
          if (cnt == 0) {
            for (int l = 0; l < nsample; ++l) {
              idx[k * nsample + l] = j;
            }
          }
          idx[k * nsample + cnt] = j;
          ++cnt;
          if (cnt >= nsample) break;
        }

    }

  }
}

void voxel_ball_query_kernel_launcher(int b, int n, int m, float min_radius, float max_radius,
                                      int nsample, const float *new_xyz, const float *xyz,
                                      int *idx, cudaStream_t stream) {
  // new_xyz: (B, M, 3)
  // xyz: (B, N, 3)
  // output:
  //      idx: (B, M, nsample)
  // b: batch_size  n: num_points  m: num_new_points

  // M的值不会超过2048.

  cudaError_t err;
  int max_threads_per_row = (SM_NUM * TOTAL_THREADS) / b;

  //每行最多能用多少块
  int max_available = DIVUP(max_threads_per_row, THREADS_PER_BLOCK) - 1;
  //每行最少需要多少块
  int min_needed = DIVUP(m, THREADS_PER_BLOCK);

  int blocks_per_row = max_available > min_needed ? min_needed : max_available;

  dim3 blocks(blocks_per_row, b);


  //dim3 blocks(DIVUP(m, THREADS_PER_BLOCK),
  //            b);  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);
  int block_size = blocks_per_row * THREADS_PER_BLOCK;

  voxel_ball_query_kernel<<<blocks, threads, 0, stream>>>(b, n, m, min_radius, max_radius,
                                                          nsample, block_size, new_xyz, xyz, idx);

  cudaDeviceSynchronize();  // for using printf in kernel function
  //printf("b: %d n: %d m: %d\n", b, n, m);
  //printf("block_size: %d blocks_per_row: %d\n", block_size, blocks_per_row);
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}
