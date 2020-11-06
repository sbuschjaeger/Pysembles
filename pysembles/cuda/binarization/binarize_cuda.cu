#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdio.h>
#include <cstdint>

#include <curand.h>
#include <curand_kernel.h>
#include <chrono>

#define DEBUG_1D 0
#define DEBUG_THREAD_INFO_FLOAT32 0
#define DEBUG_THREAD_INFO_INT32 0
#define DEBUG_BITS 0
#define DEBUG_SEEDS 0

template <typename scalar_t>
__global__ void binarize_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> input) {

  // handle access indices
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int d = blockIdx.y * blockDim.y + threadIdx.y;
  const int e = blockIdx.z * blockDim.z + threadIdx.z;

  // Python version
  //output[input > 0] = 1
  //output[input <= 0] = -1
    if ((c < input.size(0)) && (d < input.size(1)) && (e < input.size(2))) {
        if (input[c][d][e] > 0) {
            input[c][d][e] = 1;
        } else {
            input[c][d][e] = -1;
        }
    }
}

std::vector<torch::Tensor> binarize_cuda(torch::Tensor input) {
  // The number of thread blocks in a grid is usually dictated by the size of the data being processed, which typically exceeds the number of processors in the system.
  // dim3 threadsPerBlock(8,8,8)
  // <<<number of blocks per grid, number of threads ber block>>>
  // grid is created with enough blocks to have one thread per matrix element

  // https://devtalk.nvidia.com/default/topic/1028226/how-many-concurrent-threads-are-running-on-my-geforce-gtx-1080-ti-/
  const int input_size_x = input.size(0);
  const int input_size_y = input.size(1);
  const int input_size_z = input.size(2);
  int threads_x = 8; // per block, 8
  int threads_y = 8; // per block, 8
  int threads_z = 8; // per block, 8

  #if DEBUG_1D
    threads_x = 1;
    threads_y = 1;
    threads_z = 1;
  #endif

  const dim3 threads(threads_x,threads_y, threads_z);
  const dim3 blocks((input_size_x + threads_x - 1) / threads_x,
                    (input_size_y + threads_y - 1) / threads_y,
                    (input_size_z + threads_z - 1) / threads_z);

  AT_DISPATCH_ALL_TYPES(input.type(), "binarize_cuda", ([&] {
    binarize_kernel<scalar_t><<<blocks, threads>>>(
        input.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>()
    );
  }));

  return {input};
}