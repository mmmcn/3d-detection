#include <stdio.h>
#include <stdlib.h>

#define TOTAL_THREADS 2048
#define THREADS_PER_BLOCK 256
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
#define SM_NUM 40

inline int opt_n_threads(int b) {
  const int max_threads_per_block = (SM_NUM * TOTAL_THREADS) / b;
  const int pow_2 = std::log(static_cast<double>(max_threads_per_block)) / std::log(2.0);

  return max(min(1 << pow_2, TOTAL_THREADS), 1);

}

__device__ void __update(float *__restrict__ dists, int *__restrict__ dists_i,
                         int idx1, int idx2) {
  const float v1 = dists[idx1], v2 = dists[idx2];
  const int i1 = dists_i[idx1], i2 = dists_i[idx2];
  dists[idx1] = max(v1, v2);
  dists_i[idx1] = v2 > v1 ? i2 : i1;
}

template <unsigned int block_size>
__global__ void voxel_furthest_point_sampling_kernel(
    int b, int n, int m, const float *__restrict__ weights,
    const float *__restrict__ dataset,
    float *__restrict__ temp, int *__restrict__ idxs) {
    // dataset: (B, num_voxel, num_points, 3)
    // tmp: (B, num_voxel, num_points)
    // output:
    //      idx: (B, num_voxel, m)

    if (m <= 0) return;
    __shared__ float dists[block_size];
    __shared__ int dists_i[block_size];

    //int batch_index = blockIdx.y;
    //int voxel_index = blockIdx.x;

    //int index = batch_index * gridDim.x + voxel_index;

    int index = blockIdx.x;

    dataset += index * n * 3;
    temp += index * n;
    idxs += index * m;

    int tid = threadIdx.x;
    const int stride = block_size;

    //get the manhattan weights
    float weight_x = weights[0];
    float weight_y = weights[1];
    float weight_z = weights[2];

    int old = 0;
    if (threadIdx.x == 0) idxs[0] = old;

    __syncthreads();
    for (int j = 1; j < m; j++) {
        int besti = 0;
        float best = -1;

        // farthest point
        float x1 = dataset[old * 3 + 0];
        float y1 = dataset[old * 3 + 1];
        float z1 = dataset[old * 3 + 2];
        for (int k = tid; k < n; k += stride) {
            float x2, y2, z2;
            x2 = dataset[k * 3 + 0];
            y2 = dataset[k * 3 + 1];
            z2 = dataset[k * 3 + 2];

            // 遇到填充点直接跳过
            // 保证填充点不影响采样结果
            float d;
            //if (x2 == 0 && y2 == 0 && z2 == 0) {
            //    d = 0.0;
            //}
            //else{
            //    d = weight_x * (x2 - x1) * (x2 - x1) + weight_y * (y2 - y1) * (y2 - y1) + weight_z *
            //    (z2 - z1) * (z2 - z1);
            //}
            d = weight_x * (x2 - x1) * (x2 - x1) + weight_y * (y2 - y1) * (y2 - y1) + weight_z *
                (z2 - z1) * (z2 - z1);
            float d2 = min(d, temp[k]);
            temp[k] = d2;
            besti = d2 > best ? k : besti;
            best = d2 > best ? d2 : best;
        }

        // 找到了当前线程处理的点中最合适的点.
        dists[tid] = best;
        dists_i[tid] = besti;
        __syncthreads();

        if (block_size >= 1024) {
            if (tid < 512) {
                __update(dists, dists_i, tid, tid + 512);
            }
            __syncthreads();
        }
        if (block_size >= 512) {
            if (tid < 256) {
                __update(dists, dists_i, tid, tid + 256);
            }
            __syncthreads();
        }
        if (block_size >= 256) {
            if (tid < 128) {
                __update(dists, dists_i, tid, tid + 128);
            }
            __syncthreads();
        }
        if (block_size >= 128) {
            if (tid < 64) {
                __update(dists, dists_i, tid, tid + 64);
            }
            __syncthreads();
        }
        if (block_size >= 64) {
            if (tid < 32) {
                __update(dists, dists_i, tid, tid + 32);
            }
            __syncthreads();
        }
        if (block_size >= 32) {
            if (tid < 16) {
                __update(dists, dists_i, tid, tid + 16);
            }
            __syncthreads();
        }
        if (block_size >= 16) {
            if (tid < 8) {
                __update(dists, dists_i, tid, tid + 8);
            }
            __syncthreads();
        }
        if (block_size >= 8) {
            if (tid < 4) {
                __update(dists, dists_i, tid, tid + 4);
            }
            __syncthreads();
        }
        if (block_size >= 4) {
            if (tid < 2) {
                __update(dists, dists_i, tid, tid + 2);
            }
            __syncthreads();
        }
        if (block_size >= 2) {
            if (tid < 1) {
                __update(dists, dists_i, tid, tid + 1);
            }
            __syncthreads();
        }

        // dists_i[0]存放了最终得到的采样点.
        old = dists_i[0];
        if (tid == 0)
            idxs[j] = old;
    }
}

void voxel_furthest_point_sampling_kernel_launcher(int b, int n, int m,
                                                   const float *dataset,
                                                   const float *weights,
                                                   float *temp, int *idxs,
                                                   cudaStream_t stream) {
    cudaError_t err;
    unsigned int n_threads = opt_n_threads(b);
    //dim3 grid(b, num_voxel, 1);
    // x 代表有多少列, y代表有多少行
    //dim3 grid(num_voxel, b, 1);

    switch (n_threads) {
        case 2048:
            voxel_furthest_point_sampling_kernel<2048>
                <<<b, n_threads, 0, stream>>>(b, n, m, weights, dataset, temp, idxs);
            break;
        case 1024:
            voxel_furthest_point_sampling_kernel<1024>
                <<<b, n_threads, 0, stream>>>(b, n, m, weights, dataset, temp, idxs);
            break;
        case 512:
            voxel_furthest_point_sampling_kernel<512>
                <<<b, n_threads, 0, stream>>>(b, n, m, weights, dataset, temp, idxs);
            break;
        case 256:
            voxel_furthest_point_sampling_kernel<256>
                <<<b, n_threads, 0, stream>>>(b, n, m, weights, dataset, temp, idxs);
            break;
        case 128:
            voxel_furthest_point_sampling_kernel<128>
                <<<b, n_threads, 0, stream>>>(b, n, m, weights, dataset, temp, idxs);
            break;
        case 64:
            voxel_furthest_point_sampling_kernel<64>
                <<<b, n_threads, 0, stream>>>(b, n, m, weights, dataset, temp, idxs);
            break;
        case 32:
            voxel_furthest_point_sampling_kernel<32>
                <<<b, n_threads, 0, stream>>>(b, n, m, weights, dataset, temp, idxs);
            break;
        case 16:
            voxel_furthest_point_sampling_kernel<16>
                <<<b, n_threads, 0, stream>>>(b, n, m, weights, dataset, temp, idxs);
            break;
        case 8:
            voxel_furthest_point_sampling_kernel<8>
                <<<b, n_threads, 0, stream>>>(b, n, m, weights, dataset, temp, idxs);
            break;
        case 4:
            voxel_furthest_point_sampling_kernel<4>
                <<<b, n_threads, 0, stream>>>(b, n, m, weights, dataset, temp, idxs);
            break;
        case 2:
            voxel_furthest_point_sampling_kernel<2>
                <<<b, n_threads, 0, stream>>>(b, n, m, weights, dataset, temp, idxs);
            break;
        case 1:
            voxel_furthest_point_sampling_kernel<1>
                <<<b, n_threads, 0, stream>>>(b, n, m, weights, dataset, temp, idxs);
            break;
        default:
            voxel_furthest_point_sampling_kernel<512>
                <<<b, n_threads, 0, stream>>>(b, n, m, weights, dataset, temp, idxs);
    }

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
