#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define N 512
#define K 64

__global__ void printf_base(){
	printf("hello\n");
	
}

void printf_base_test()
{
	dim3 block(1);
	dim3 grid(1);
	
	printf_base<<<grid, block>>>();
}

__global__ void grid_block_Idx()
{
	printf("blockIdx.x=%d blockIdx.y=%d threadIdx.x=%d threadIdx.y=%d kernel print test\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}


void grid_block_Idx_test()
{
	//’Î∂‘block»›¡ø≤‚ ‘
	dim3 block(32, 32);
	
	//’Î∂‘grid»›¡ø≤‚ ‘
	dim3 grid(32, 32);
	
	printf("grid.x=%d grid.y=%d grid.z=%d\n", grid.x, grid.y, grid.z);
	printf("block.x=%d block.y=%d block.z=%d\n", block.x, block.y, block.z);
	printf("-------------------------------\n");
}

int main()
{
	printf_base_test();
	grid_block_Idx_test();
	return 0;
}