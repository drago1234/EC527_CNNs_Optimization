#include <stdio.h>

#include <cstdio>
#include <cstdlib>
#include <math.h>


// Goal: Comparing the difference between two mul operation: * and IMUL(a, b)
void initializeArray1D(float *arr, int len, int seed);


#define IMUL(a, b) __mul24(a, b)
// Purpose: In this example we are comparing the whether the given new multiplication operation IMUL(a, b) in lab7 is the same as "*" operation.
__global__ void mul(int a, int b, int *c){
	*c = IMUL(a, b);
	printf("Result of IMUL(2, 7) in GPU = %d\n", *c);
}

void call_mul(void){
	printf("====================> Running on call_mul\n");

	// Running multiplication on CPU
	printf("Result of 2*7 in CPU = %d\n", 2*7);

	// Running multiplication on GPU
	int c;
	int *device_c;
	cudaMalloc((void**)&device_c, sizeof(int));
	mul<<<1, 1>>>(2, 7, device_c);
	cudaMemcpy(&c, device_c, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(device_c);

	// Print out the result computed from GPU
	printf("Result of IMUL(2, 7) in GPU = %d\n", c);
}



#include "cuPrintf.cu"
#include "cuPrintf.cuh"
// Purpose: we are testing how is the cuPrintf() function given in Lab7 works:
__global__ void test_cuPrintf(void){
	cuPrintf("Print some value: %f\n", 666);
}
void call_test_cuPrintf(void){
	printf("====================> Running on call_test_cuPrintf\n");
	// Running multiplication on CPU
	printf("We are expected to see some value get print out by cuPrintf() that running on GPU!\n");
	cudaPrintfInit();
	test_cuPrintf<<<1, 1>>>();
	cudaPrintfDisplay(stdout, true); cudaPrintfEnd();
}


/* 
Purpose: We created three kernel functions to test different way of locating thread's position. All execution parameter configuration in 1D.
1) c[bId_x] = a[bId_x] + b[bId_x]; ==> vector_add_combine_threads_and_blocks<<<N, 1>>>(dev_a, dev_b, dev_e);

2) c[tId_x] = a[tId_x] + b[tId_x];	==> vector_add_with_threads <<<1, N>>> (dev_a, dev_b, dev_d);

3)  i = threadIdx.x + blockDim.x * blockIdx.x; c[i] = a[i] + b[i]; ==> vector_add_with_threads <<<1, N>>> (dev_a, dev_b, dev_d);

In the end, we will see that method 3 is the best practice!
*/
#define N 16	// Vctor length
// Version using block
__global__ void vector_add_with_blocks(int *a, int *b, int *c){
	int bId_x = blockIdx.x;	/// handle the data at this index
	if (bId_x < N){
		c[bId_x] = a[bId_x] + b[bId_x];
	}
	cuPrintf("Print some value: %f\n", c[bId_x]);
}

// Version using threads in one block
__global__ void vector_add_with_threads(int *a, int *b, int *c){
	int tId_x = threadIdx.x;	/// handle the data at this index
	if (tId_x < N){
		c[tId_x] = a[tId_x] + b[tId_x];
	}
	cuPrintf("Print some value: %f\n", c[tId_x]);	// You won't see the result by putting cuPrintf() here
}

__global__ void vector_add_combine_threads_and_blocks(int *a, int *b, int *c){
	int i = threadIdx.x + blockDim.x * blockIdx.x;	/// handle the data at this index
	if (i < N){
		c[i] = a[i] + b[i];
	}
	cuPrintf("Print some value: %f\n", c[i]);	// You won't see the result by putting cuPrintf() here
}

void call_vector_add(void){
	printf("====================> Running on call_vector_add\n");
	int count, i;

	// Find number of GPUs
	cudaGetDeviceCount(&count);
	("There are %d GPU devices in your system\n", count);

	int a[N], b[N], c[N], d[N], e[N], i;
	int *dev_a, *dev_b, *dev_c, *dev_d, *dev_e;

	// Allocate GPU memory
	cudaMalloc((void**) &dev_a, N*sizeof(int));
	cudaMalloc((void**) &dev_b, N*sizeof(int));
	cudaMalloc((void**) &dev_c, N*sizeof(int));
	cudaMalloc((void**) &dev_d, N*sizeof(int));
	cudaMalloc((void**) &dev_e, N*sizeof(int));

	// Allocate arrays on host memory and initialize them
	for (i=0; i<N; i++){
		a[i] = i;
		b[i] = i*i;
	}

	// Transfer the arrays to the GPU memory
	cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);

	// Launch the kernel function on GPU
	// 1D vector on multiple block, with shape N block, and 1 thread each block
	cudaPrintfInit();
	printf("Start running on vector_add_with_blocks, with dimGrid(%d), dimBlock(%d)\n", N, 1);
	vector_add_with_blocks<<<N, 1>>>(dev_a, dev_b, dev_c);

	/*
		Note: for vector_add_with_threads(), --> We are trying to use all threads in one blocks to perform the computation, so you should declared your kernel with shape <<<1, N>>> instead of <<<N, 1>>>
	*/
	printf("Start running on vector_add_with_threads, with dimGrid(%d), dimBlock(%d)\n", N, 1);
	vector_add_with_threads<<<1, N>>>(dev_a, dev_b, dev_d);

	printf("Start running on vector_add_with_threads, with dimGrid(%d), dimBlock(%d)\n", N, 1);
	vector_add_combine_threads_and_blocks<<<N, 1>>>(dev_a, dev_b, dev_e);

	cudaPrintfDisplay(stdout, true); cudaPrintfEnd();

	cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(d, dev_d, N*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(e, dev_e, N*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);cudaFree(dev_d); cudaFree(dev_e);

	// Print out the result computed from GPU
	printf("a[i] + b[i] = c[i] = d[i] = e[i], where c[i] is the result running with blockIdx.x, and d[i] is the result running with threadIdx.x, and e[i] is the result running with threadIdx.x\n");
	for(i=0;i<N;i++){
		printf("%d + %d = %d = %d = %d\n", a[i], b[i], c[i], d[i], e[i]);
	}
}


__global__ void kernel_add (int arrLen, float* x, float* y, float* result) {
  const int tid = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
  const int threadN = IMUL(blockDim.x, gridDim.x);  // Number of thread per grid

  int i;
  // i = blockDim.x*blockIdx.x + threadIdx.x;
  // if(i<arrLen) result[i] = (1e-6 * x[i] ) + (1e-7 * y[i]) + 0.25;

  for(i = tid; i < arrLen; i += threadN) {
    result[i] = (1e-6 * x[i] ) + (1e-7 * y[i]) + 0.25;
  }
}



int main(void){
	printf("\n\n");		// Space out from previous output!
	call_vector_add();
	// call_test_cuPrintf();
	// call_mul();

	return 0;
}